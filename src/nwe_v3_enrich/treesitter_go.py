"""
Tree-sitter based Go AST parsing for accurate chunk classification.

This module replaces regex-based detection with proper AST parsing to:
- Eliminate misclassifications (method vs function)
- Provide accurate primary_symbol and ast_path
- Support multi-declaration chunks
- Handle method bodies correctly via byte span overlap
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import re

try:
    from tree_sitter_languages import get_language, get_parser
    GO_LANG = get_language('go')
    HAS_TREESITTER = True
except ImportError:
    HAS_TREESITTER = False
    GO_LANG = None


@dataclass
class TSNodeInfo:
    """Information about a top-level Go declaration."""
    kind: str  # method, function, type
    start_byte: int
    end_byte: int
    name: Optional[str] = None
    receiver: Optional[str] = None
    type_name: Optional[str] = None
    type_kind: Optional[str] = None  # struct, interface, alias


@dataclass
class TSFileIndex:
    """Parsed file index with all top-level declarations."""
    source_bytes: bytes
    tree: Any
    top_level: List[TSNodeInfo]
    package_name: Optional[str]
    imports: List[str]


class GoTSIndexer:
    """Tree-sitter based Go source indexer."""
    
    def __init__(self):
        if not HAS_TREESITTER:
            raise ImportError("tree_sitter_languages not available")
        self.parser = get_parser('go')
    
    def parse_file(self, source: str) -> TSFileIndex:
        """
        Parse Go source file and build index of top-level declarations.
        
        Args:
            source: Go source code as string
            
        Returns:
            TSFileIndex with all top-level declarations
        """
        src_b = source.encode('utf-8')
        tree = self.parser.parse(src_b)
        root = tree.root_node
        
        package_name = None
        imports = []
        top_level: List[TSNodeInfo] = []
        
        for child in root.children:
            k = child.type
            
            if k == 'package_clause':
                package_name = self._extract_package_name(child, src_b)
            
            elif k == 'import_declaration':
                imports.extend(self._collect_imports(child, src_b))
            
            elif k == 'function_declaration':
                # Check if this is actually a method (has receiver)
                recv = self._extract_receiver(child, src_b)
                fn_name = self._extract_function_name(child, src_b)
                
                if recv:
                    # Method declaration
                    type_name = self._receiver_type_name(recv)
                    top_level.append(TSNodeInfo(
                        kind='method',
                        start_byte=child.start_byte,
                        end_byte=child.end_byte,
                        name=fn_name,
                        receiver=recv,
                        type_name=type_name,
                        type_kind=None,
                    ))
                else:
                    # Regular function
                    top_level.append(TSNodeInfo(
                        kind='function',
                        start_byte=child.start_byte,
                        end_byte=child.end_byte,
                        name=fn_name,
                        receiver=None,
                        type_name=None,
                        type_kind=None,
                    ))
            
            elif k == 'method_declaration':
                # Explicit method declaration node
                fn_name = self._extract_function_name(child, src_b)
                recv = self._extract_receiver(child, src_b)
                type_name = self._receiver_type_name(recv) if recv else None
                
                top_level.append(TSNodeInfo(
                    kind='method',
                    start_byte=child.start_byte,
                    end_byte=child.end_byte,
                    name=fn_name,
                    receiver=recv,
                    type_name=type_name,
                ))
            
            elif k == 'type_declaration':
                # May contain multiple type_spec children
                for spec in self._iter_type_specs(child):
                    tname, tkind = self._type_spec_info(spec, src_b)
                    top_level.append(TSNodeInfo(
                        kind='type',
                        start_byte=spec.start_byte,
                        end_byte=spec.end_byte,
                        name=tname,
                        type_name=tname,
                        type_kind=tkind,
                    ))
        
        return TSFileIndex(src_b, tree, top_level, package_name, imports)
    
    def locate_for_chunk(self, index: TSFileIndex, start_byte: int, end_byte: int) -> Dict[str, Any]:
        """
        Map a chunk to owning symbol(s) based on byte span overlap.
        
        Args:
            index: Pre-parsed file index
            start_byte: Chunk start position
            end_byte: Chunk end position
            
        Returns:
            Dictionary with node_kind, primary_symbol, ast_path, etc.
        """
        # Find all top-level nodes overlapping the chunk
        overlaps = [
            n for n in index.top_level 
            if not (end_byte <= n.start_byte or start_byte >= n.end_byte)
        ]
        
        result: Dict[str, Any] = {
            "node_kind": "unknown",
            "primary_symbol": "",
            "receiver": "",
            "method_name": "",
            "function_name": "",
            "type_name": "",
            "type_kind": "",
            "ast_path": "go:block",
            "is_header": False,
            "def_symbols": [],
        }
        
        if overlaps:
            # PHASE 2a FIX: Sort by source order (start_byte) for stable, predictable arrays
            overlaps_sorted = sorted(overlaps, key=lambda n: n.start_byte)
            
            # Calculate overlap size for each node to find primary
            def calc_overlap(n):
                return min(end_byte, n.end_byte) - max(start_byte, n.start_byte)
            
            # Primary = largest overlap (for backward compatibility)
            primary = max(overlaps_sorted, key=calc_overlap)
            primary_idx = overlaps_sorted.index(primary)
            
            self._fill_from_tsnode(primary, result, start_byte, end_byte)
            
            # Multi-declaration detection and metadata
            is_multi = len(overlaps_sorted) > 1
            result["is_multi_declaration"] = is_multi
            
            if is_multi:
                # Build parallel arrays for all overlapping declarations
                all_symbols = []
                all_kinds = []
                all_ast_paths = []
                all_roles = []
                all_receivers = []
                all_type_names = []
                all_type_kinds = []
                all_start_bytes = []
                all_end_bytes = []
                
                for n in overlaps_sorted:
                    # Format metadata for this node (non-mutating)
                    node_meta = self._format_tsnode(n, start_byte, end_byte)
                    
                    all_symbols.append(node_meta["symbol"])
                    all_kinds.append(node_meta["kind"])
                    all_ast_paths.append(node_meta["ast_path"])
                    all_roles.append(node_meta["role"])
                    all_receivers.append(node_meta.get("receiver", ""))
                    all_type_names.append(node_meta.get("type_name", ""))
                    all_type_kinds.append(node_meta.get("type_kind", ""))
                    # ISSUE 1 FIX: Add per-symbol byte ranges
                    all_start_bytes.append(n.start_byte)
                    all_end_bytes.append(n.end_byte)
                
                result["all_symbols"] = all_symbols
                result["all_kinds"] = all_kinds
                result["all_ast_paths"] = all_ast_paths
                result["all_roles"] = all_roles
                result["all_receivers"] = all_receivers
                result["all_type_names"] = all_type_names
                result["all_type_kinds"] = all_type_kinds
                result["all_start_bytes"] = all_start_bytes
                result["all_end_bytes"] = all_end_bytes
                # PHASE 2b: Add normalized receivers for consistency with ast_path
                result["all_receivers_normalized"] = [self._normalize_receiver(r) for r in all_receivers]
                # PHASE 2a FIX: primary_index points to primary in source-ordered array
                result["primary_index"] = primary_idx
            else:
                # Single declaration - populate arrays with single item
                # ISSUE 2 FIX: Use proper role for types
                role = self._determine_chunk_role(primary, start_byte, end_byte)
                if primary.kind == "type" and not role:
                    role = "declaration"
                
                result["all_symbols"] = [result["primary_symbol"]]
                result["all_kinds"] = [result["node_kind"]]
                result["all_ast_paths"] = [result["ast_path"]]
                result["all_roles"] = [role]
                result["all_receivers"] = [result.get("receiver", "")]
                result["all_type_names"] = [result.get("type_name", "")]
                # PHASE 2b: Use None for non-types instead of empty string
                result["all_type_kinds"] = [result.get("type_kind") if result.get("type_kind") else None]
                # ISSUE 1 FIX: Add byte ranges for single declaration
                result["all_start_bytes"] = [primary.start_byte]
                result["all_end_bytes"] = [primary.end_byte]
                # PHASE 2b: Add normalized receiver for single declarations
                result["all_receivers_normalized"] = [self._normalize_receiver(result.get("receiver", ""))]
                # ISSUE 5 FIX: primary_index is 0 for single declarations
                result["primary_index"] = 0
            
            # PHASE 2c: def_symbols = secondary symbols only (exclude primary)
            # Historical expectation: "other symbols" not including the primary
            primary_idx = result.get("primary_index", 0)
            result["def_symbols"] = [s for i, s in enumerate(result["all_symbols"]) 
                                     if i != primary_idx]
            
            # PHASE 2b: Validation - ensure primary_* stays in sync with all_*
            primary_idx = result.get("primary_index", 0)
            assert result["primary_symbol"] == result["all_symbols"][primary_idx], \
                f"primary_symbol mismatch: {result['primary_symbol']} != {result['all_symbols'][primary_idx]}"
            assert result["node_kind"] == result["all_kinds"][primary_idx], \
                f"primary_kind mismatch: {result['node_kind']} != {result['all_kinds'][primary_idx]}"
            
            return result
        
        # Try header inference: intersects package/imports region?
        if self._intersects_header(index, start_byte, end_byte):
            result["node_kind"] = "header"
            result["is_header"] = True
            result["ast_path"] = "go:file_header"
            return result
        
        # Fallback: find smallest enclosing declaration and bubble up
        owner = self._bubble_to_owner(index, start_byte, end_byte)
        if owner:
            self._fill_from_tsnode(owner, result, start_byte, end_byte)
            return result
        
        return result
    
    # Helper methods
    
    def _normalize_receiver(self, raw_receiver: str) -> str:
        """
        Normalize receiver format for consistency with ast_path.
        
        Examples:
          "s *Service"       → "(*Service)"
          "s Service"        → "(Service)"
          "suite *TestSuite" → "(*TestSuite)"
          ""                 → ""
        
        PHASE 2b: Added for receiver format consistency
        """
        if not raw_receiver:
            return ""
        
        # Parse "name Type" or "name *Type"
        parts = raw_receiver.split()
        if len(parts) < 2:
            return raw_receiver  # Malformed, return as-is
        
        type_part = parts[1]  # "*Service" or "Service"
        
        if type_part.startswith("*"):
            return f"(*{type_part[1:]})"
        else:
            return f"({type_part})"
    
    def _extract_package_name(self, node, src_b: bytes) -> Optional[str]:
        """Extract package name from package_clause node."""
        for child in node.children:
            if child.type == 'package_identifier':
                return src_b[child.start_byte:child.end_byte].decode('utf-8')
        return None
    
    def _collect_imports(self, node, src_b: bytes) -> List[str]:
        """Collect import paths from import_declaration node."""
        imports = []
        for child in node.children:
            if child.type == 'import_spec':
                for subchild in child.children:
                    if subchild.type == 'interpreted_string_literal':
                        import_path = src_b[subchild.start_byte:subchild.end_byte].decode('utf-8')
                        # Remove quotes
                        import_path = import_path.strip('"')
                        imports.append(import_path)
            elif child.type == 'import_spec_list':
                for spec in child.children:
                    if spec.type == 'import_spec':
                        for subchild in spec.children:
                            if subchild.type == 'interpreted_string_literal':
                                import_path = src_b[subchild.start_byte:subchild.end_byte].decode('utf-8')
                                import_path = import_path.strip('"')
                                imports.append(import_path)
        return imports
    
    def _extract_function_name(self, node, src_b: bytes) -> Optional[str]:
        """Extract function/method name."""
        # Try to get name field directly
        name_node = node.child_by_field_name('name')
        if name_node:
            return src_b[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Fallback: look for identifier children
        for child in node.children:
            if child.type == 'identifier':
                return src_b[child.start_byte:child.end_byte].decode('utf-8')
        return None
    
    def _extract_receiver(self, node, src_b: bytes) -> Optional[str]:
        """Extract receiver from function/method declaration."""
        # Try to get receiver field directly
        receiver_node = node.child_by_field_name('receiver')
        if receiver_node:
            # Extract full receiver text: "s *Service" or "s Service"
            text = src_b[receiver_node.start_byte:receiver_node.end_byte].decode('utf-8')
            # Remove parentheses
            text = text.strip('()')
            return text.strip()
        
        # No receiver field found - this is a regular function, not a method
        return None
    
    def _receiver_type_name(self, recv: str) -> Optional[str]:
        """
        Extract type name from receiver string.
        Examples: "s *Service" -> "Service", "s Service" -> "Service"
        """
        if not recv:
            return None
        
        # Handle patterns: "s *Service", "s *pkg.Service", "s Service"
        parts = recv.split()
        if len(parts) >= 2:
            # Last part is the type (may include pointer and package)
            type_part = parts[-1]
            # Remove pointer if present
            type_part = type_part.lstrip('*')
            # If package-qualified, take last segment
            if '.' in type_part:
                type_part = type_part.split('.')[-1]
            return type_part
        
        return None
    
    def _iter_type_specs(self, type_decl_node):
        """Iterate over type_spec children in type_declaration."""
        for child in type_decl_node.children:
            if child.type == 'type_spec':
                yield child
    
    def _type_spec_info(self, spec, src_b: bytes) -> Tuple[str, Optional[str]]:
        """
        Extract type name and kind from type_spec node.
        Returns: (type_name, type_kind) where kind is struct/interface/alias/None
        """
        type_name = None
        type_kind = None
        
        # Try to get name field directly
        name_node = spec.child_by_field_name('name')
        if name_node and name_node.type == 'type_identifier':
            type_name = src_b[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Try to get type field directly
        type_node = spec.child_by_field_name('type')
        if type_node:
            if type_node.type == 'struct_type':
                type_kind = 'struct'
            elif type_node.type == 'interface_type':
                type_kind = 'interface'
            elif type_node.type in ('type_identifier', 'qualified_type'):
                type_kind = 'alias'
        
        # Fallback: iterate children
        if not type_kind:
            for child in spec.children:
                if child.type == 'struct_type':
                    type_kind = 'struct'
                elif child.type == 'interface_type':
                    type_kind = 'interface'
                elif child.type in ('type_identifier', 'qualified_type') and not type_name:
                    # This might be an alias
                    type_kind = 'alias'
        
        return type_name or "", type_kind
    
    def _intersects_header(self, index: TSFileIndex, start_byte: int, end_byte: int) -> bool:
        """Check if chunk intersects package/import region."""
        # Find the byte range of package and imports
        header_end = 0
        first_decl_start = None
        
        root = index.tree.root_node
        for child in root.children:
            # Skip whitespace/newlines
            if not child.type.strip():
                continue
            
            if child.type in ('package_clause', 'import_declaration', 'comment'):
                header_end = max(header_end, child.end_byte)
            elif child.type in ('function_declaration', 'method_declaration', 'type_declaration'):
                # First real declaration marks end of header
                if first_decl_start is None:
                    first_decl_start = child.start_byte
                break
        
        # If no header found, return False
        if header_end == 0:
            return False
        
        # Use first declaration start as header boundary if available
        if first_decl_start is not None:
            header_end = max(header_end, first_decl_start - 1)
        
        # Check if chunk is primarily in header region
        # Chunk must start before first non-header element and overlap significantly
        if start_byte >= header_end:
            return False
        
        overlap = min(end_byte, header_end) - max(start_byte, 0)
        chunk_size = end_byte - start_byte
        
        # More lenient: >30% overlap with header OR chunk ends before any declarations
        return (overlap > chunk_size * 0.3) or (end_byte <= header_end)
    
    def _bubble_to_owner(self, index: TSFileIndex, start_byte: int, end_byte: int) -> Optional[TSNodeInfo]:
        """
        Find the smallest enclosing declaration that contains this chunk.
        Used for method/function body chunks.
        """
        # Find declarations that fully contain this chunk
        containers = [
            n for n in index.top_level
            if n.start_byte <= start_byte and end_byte <= n.end_byte
        ]
        
        if containers:
            # Return the smallest container (tightest fit)
            return min(containers, key=lambda n: n.end_byte - n.start_byte)
        
        return None
    
    def _compact_def(self, n: TSNodeInfo) -> Dict[str, str]:
        """Create compact representation for def_symbols."""
        return {
            "name": n.name or "",
            "kind": n.kind,
            "type": n.type_name or ""
        }
    
    def _format_tsnode(self, n: TSNodeInfo, start_byte: int, end_byte: int) -> Dict[str, Any]:
        """
        Pure formatter: convert TSNodeInfo to metadata dict without mutation.
        Used for building parallel arrays in multi-declaration chunks.
        
        Returns:
            Dict with keys: symbol, kind, ast_path, role, receiver, type_name, type_kind
        """
        meta: Dict[str, Any] = {
            "symbol": n.name or "",
            "kind": n.kind,
            "role": "",
            "ast_path": "",
            "receiver": "",
            "type_name": "",
            "type_kind": None,  # PHASE 2b: Use None instead of "" for non-types
        }
        
        # Determine chunk role (signature/body/mixed)
        chunk_role = self._determine_chunk_role(n, start_byte, end_byte)
        # ISSUE 2 FIX: Types should have role="declaration", not empty string
        if n.kind == "type" and not chunk_role:
            chunk_role = "declaration"
        meta["role"] = chunk_role
        
        if n.kind == "method":
            meta["receiver"] = n.receiver or ""
            meta["type_name"] = n.type_name or ""
            
            # Format receiver for ast_path
            if n.receiver and "*" in n.receiver:
                recv_norm = f"(*{n.type_name})"
            else:
                recv_norm = f"({n.type_name})"
            
            # Build ast_path with role suffix
            base_path = f"go:method:{recv_norm}.{n.name}"
            meta["ast_path"] = f"{base_path}/{chunk_role}" if chunk_role else base_path
        
        elif n.kind == "function":
            base_path = f"go:function:{n.name}" if n.name else "go:function:unknown"
            meta["ast_path"] = f"{base_path}/{chunk_role}" if chunk_role else base_path
        
        elif n.kind == "type":
            tk = n.type_kind or "unknown"
            meta["type_name"] = n.type_name or n.name or ""
            meta["type_kind"] = tk
            tn = meta["type_name"] or "unknown"
            meta["ast_path"] = f"go:type:{tn} ({tk})"
        
        else:
            meta["ast_path"] = "go:block"
        
        return meta
    
    def _fill_from_tsnode(self, n: TSNodeInfo, result: Dict[str, Any], start_byte: int = 0, end_byte: int = 0) -> None:
        """Fill result dictionary from TSNodeInfo."""
        result["node_kind"] = n.kind if n.kind in {"method", "function", "type"} else "unknown"
        
        # Determine chunk role (signature vs body)
        chunk_role = self._determine_chunk_role(n, start_byte, end_byte)
        
        if n.kind == "method":
            result["receiver"] = n.receiver or ""
            result["method_name"] = n.name or ""
            result["type_name"] = n.type_name or ""
            
            # Format receiver for ast_path
            if n.receiver and "*" in n.receiver:
                recv_norm = f"(*{n.type_name})"
            else:
                recv_norm = f"({n.type_name})"
            
            # Add chunk role to ast_path for disambiguation
            base_path = f"go:method:{recv_norm}.{n.name}"
            result["ast_path"] = f"{base_path}/{chunk_role}" if chunk_role else base_path
            result["primary_symbol"] = n.name or ""
        
        elif n.kind == "function":
            result["function_name"] = n.name or ""
            base_path = f"go:function:{n.name}" if n.name else "go:function:unknown"
            result["ast_path"] = f"{base_path}/{chunk_role}" if chunk_role else base_path
            result["primary_symbol"] = n.name or ""
        
        elif n.kind == "type":
            tk = n.type_kind or "unknown"
            result["type_name"] = n.type_name or n.name or ""
            result["type_kind"] = tk
            tn = result["type_name"] or "unknown"
            result["ast_path"] = f"go:type:{tn} ({tk})"
            result["primary_symbol"] = result["type_name"]
    
    def _determine_chunk_role(self, n: TSNodeInfo, start_byte: int, end_byte: int) -> str:
        """
        Determine chunk role for methods/functions.
        Returns: "signature", "body", "mixed", "complete", or "" (for types/unknown)
        
        PHASE 2b: Fixed terminology - "complete" for full functions, "declaration" only for types
        
        Roles:
          - "complete": Chunk covers >85% of function/method (nearly/fully complete)
          - "signature": Small chunk with just signature (<15%)
          - "mixed": Starts at signature but partial coverage (15-85%)
          - "body": Chunk is deep in body (no signature)
          - "": Not applicable (types use "declaration" elsewhere)
        """
        if n.kind not in {"method", "function"}:
            return ""
        
        # Calculate metrics
        decl_size = n.end_byte - n.start_byte
        chunk_start_offset = start_byte - n.start_byte
        chunk_size = end_byte - start_byte
        chunk_coverage_pct = (chunk_size / decl_size) * 100 if decl_size > 0 else 100
        
        # PHASE 2b: If chunk covers >85% of function/method, it's complete
        if chunk_coverage_pct > 85:
            return "complete"
        
        # If chunk starts near the beginning (within 30 bytes of signature)
        if chunk_start_offset < 30:
            # Signature-only: small chunk covering less than 15% of declaration
            if chunk_coverage_pct < 15 and chunk_size < 200:
                return "signature"
            # Mixed: starts at beginning but doesn't cover most of declaration
            elif chunk_coverage_pct <= 85:
                return "mixed"
            else:
                return "complete"
        else:
            # Body: chunk starts well into the declaration (past signature)
            return "body"


# Module-level singleton
_indexer = None

def get_indexer() -> GoTSIndexer:
    """Get or create singleton indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = GoTSIndexer()
    return _indexer

