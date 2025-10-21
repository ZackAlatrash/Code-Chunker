import os
import subprocess
from typing import List, Dict, Union, Tuple
import sys
import tree_sitter  # for __version__
from tree_sitter import Language, Parser, Node
import logging
from importlib.metadata import version as pkg_version, PackageNotFoundError
def _platform_ext() -> str:
    if sys.platform.startswith("win"):
        return ".dll"
    elif sys.platform == "darwin":
        return ".so"  # works fine on mac to emit .so
    else:
        return ".so"

def _abi_tag() -> str:
    abi = getattr(Language, "ABI_VERSION", None)
    ts_ver = getattr(tree_sitter, "__version__", "unknown")
    return f"abi{abi or 'NA'}-ts{ts_ver}"

class CodeParser:
    # Added a CACHE_DIR class attribute for caching
    CACHE_DIR = os.path.expanduser("~/.code_parser_cache")

    def __init__(self, file_extensions: Union[None, List[str], str] = None):
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]
        self.language_extension_map = {
            "py": "python",
            "js": "javascript",
            "jsx": "javascript",
            "css": "css",
            "ts": "typescript",
            # Use dedicated TSX grammar name when using prebuilt grammars
            "tsx": "tsx",
            "php": "php",
            "rb": "ruby",
            "go": "go"
        }
        if file_extensions is None:
            self.language_names = []
        else:
            self.language_names = [self.language_extension_map.get(ext) for ext in file_extensions if
                                   ext in self.language_extension_map]
        self.languages = {}
        self._install_parsers()

    def _install_parsers(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        try:
            # Ensure cache directory exists
            if not os.path.exists(self.CACHE_DIR):
                os.makedirs(self.CACHE_DIR)

            build_dir = os.path.join(self.CACHE_DIR, "build")
            os.makedirs(build_dir, exist_ok=True)

            ext = _platform_ext()
            tag = _abi_tag()

            # Try to use the prebuilt grammars package if available
            get_prebuilt_language = None
            try:
                from tree_sitter_languages import get_language as _get_language  # type: ignore
                get_prebuilt_language = _get_language
                logging.info("Detected tree_sitter_languages; will prefer prebuilt parsers")
            except Exception:
                get_prebuilt_language = None

            for language in self.language_names:
                # 0) Fast path: prebuilt grammar (if package present and supports this lang)
                if get_prebuilt_language is not None:
                    try:
                        self.languages[language] = get_prebuilt_language(language)
                        logging.info(f"Loaded prebuilt {language} parser")
                        continue
                    except Exception as e:
                        logging.warning(f"Prebuilt {language} parser not available ({e})")

                # If we reach here, either prebuilt is unavailable or failed for this language.
                # With tree-sitter >= 0.21, Python bindings no longer expose build_library.
                # Only attempt build fallback if the API supports it; otherwise skip.
                if not hasattr(Language, "build_library"):
                    logging.warning(
                        f"Skipping build for {language}: installed tree_sitter has no build_library. "
                        "Install tree_sitter_languages or use an older tree-sitter (<0.21) to build."
                    )
                    continue

                repo_path = os.path.join(self.CACHE_DIR, f"tree-sitter-{language}")

                # Check if the repository exists and contains necessary files
                if not os.path.exists(repo_path) or not self._is_repo_valid(repo_path, language):
                    try:
                        if os.path.exists(repo_path):
                            logging.info(f"Updating existing repository for {language}")
                            update_command = f"cd {repo_path} && git pull"
                            subprocess.run(update_command, shell=True, check=True)
                        else:
                            logging.info(f"Cloning repository for {language}")
                            clone_command = f"git clone https://github.com/tree-sitter/tree-sitter-{language} {repo_path}"
                            subprocess.run(clone_command, shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Failed to clone/update repository for {language}. Error: {e}")
                        continue

                try:
                    # Initialize per-language directories for safer error logging
                    ts_dir = tsx_dir = php_dir = None
                    # Compute a versioned output path
                    build_path = os.path.join(build_dir, f"{language}-{tag}{ext}")

                    # Clean up any legacy untagged file that could cause confusion
                    legacy_path_so = os.path.join(build_dir, f"{language}.so")
                    legacy_path_dylib = os.path.join(build_dir, f"{language}.dylib")
                    legacy_path_dll = os.path.join(build_dir, f"{language}.dll")
                    for lp in (legacy_path_so, legacy_path_dylib, legacy_path_dll):
                        if os.path.exists(lp):
                            try:
                                os.remove(lp)
                            except OSError:
                                pass

                    # Only build if the ABI-tagged file doesn't exist yet
                    if not os.path.exists(build_path):
                        # Special handling for TypeScript & PHP
                        if language == 'typescript':
                            ts_dir = os.path.join(repo_path, 'typescript')
                            tsx_dir = os.path.join(repo_path, 'tsx')
                            if os.path.exists(ts_dir) and os.path.exists(tsx_dir):
                                Language.build_library(build_path, [ts_dir, tsx_dir])
                            else:
                                raise FileNotFoundError(f"TypeScript or TSX directory not found in {repo_path}")
                        elif language == 'php':
                            php_dir = os.path.join(repo_path, 'php')
                            if os.path.exists(php_dir):
                                Language.build_library(build_path, [php_dir])
                            else:
                                raise FileNotFoundError(f"PHP directory not found in {repo_path}")
                        else:
                            Language.build_library(build_path, [repo_path])

                    # --- load the compiled language (handle old & new APIs) ---
                    lang = None
                    try:
                        # New API in 0.25.x: Language.load(path) -> Language
                        if hasattr(Language, "load"):
                            lang = Language.load(build_path)  # type: ignore[attr-defined]
                        else:
                            # Old API: Language(path, name)
                            lang = Language(build_path, language)
                    except TypeError:
                        # Some 0.25 builds removed the (path, name) ctor entirely
                        lang = Language.load(build_path)  # type: ignore[attr-defined]

                    self.languages[language] = lang
                    logging.info(f"Successfully built and loaded {language} parser from {build_path}")
                except Exception as e:
                    logging.error(f"Failed to build or load language {language}. Error: {str(e)}")
                    logging.error(f"Repository path: {repo_path}")
                    logging.error(f"Build path: {build_path}")
                    if language == 'typescript':
                        logging.error(f"TypeScript dir exists: {os.path.exists(ts_dir) if ts_dir else 'N/A'}")
                        logging.error(f"TSX dir exists: {os.path.exists(tsx_dir) if tsx_dir else 'N/A'}")
                    elif language == 'php':
                        logging.error(f"PHP dir exists: {os.path.exists(php_dir) if php_dir else 'N/A'}")

        except Exception as e:
            logging.error(f"An unexpected error occurred during parser installation: {str(e)}")

    def _is_repo_valid(self, repo_path: str, language: str) -> bool:
        """Check if the repository contains necessary files."""
        if language == 'typescript':
            return (os.path.exists(os.path.join(repo_path, 'typescript', 'src', 'parser.c')) and
                     os.path.exists(os.path.join(repo_path, 'tsx', 'src', 'parser.c')))
        elif language == 'php':
            return os.path.exists(os.path.join(repo_path, 'php', 'src', 'parser.c'))
        else:
            return os.path.exists(os.path.join(repo_path, 'src', 'parser.c'))

    def parse_code(self, code: str, file_extension: str) -> Union[None, Node]:
        language_name = self.language_extension_map.get(file_extension)
        if language_name is None:
            print(f"Unsupported file type: {file_extension}")
            return None

        language = self.languages.get(language_name)
        if language is None:
            print("Language parser not found")
            return None

        parser = Parser()
        parser.set_language(language)
        tree = parser.parse(bytes(code, "utf8"))

        if tree is None:
            print("Failed to parse the code")
            return None

        return tree.root_node

    def is_top_level_declaration(self, node: Node) -> bool:
        """
        Check if a node is a top-level declaration (not nested inside a function/method/struct).
        
        This prevents nested declarations from creating chunk boundaries:
        - Types/functions nested inside functions
        - Anonymous structs nested inside struct fields (Go)
        - Nested classes/functions (Python, JS)
        
        Args:
            node: Tree-sitter node to check
            
        Returns:
            True if node is at top level (file/module scope), False if nested
        """
        parent = node.parent
        while parent:
            # If we find a function or method declaration in the parent chain,
            # this node is nested inside it
            if parent.type in [
                'function_declaration',
                'method_declaration',
                'function_definition',  # Python
                'arrow_function',       # JavaScript/TypeScript
                'function_item',        # Rust
            ]:
                return False
            
            # If this is a struct/interface nested inside a field declaration,
            # reject it (common in Go with anonymous struct fields)
            if parent.type in ['field_declaration', 'field_declaration_list']:
                return False
            
            # If this struct_type is nested inside another struct_type, reject it
            # (handles deeply nested anonymous structs)
            if node.type in ['struct_type', 'interface_type'] and parent.type in ['struct_type', 'interface_type']:
                return False
            
            parent = parent.parent
        return True

    def extract_points_of_interest(self, node: Node, file_extension: str) -> List[Tuple[Node, str]]:
        node_types_of_interest = self._get_node_types_of_interest(file_extension)

        points_of_interest = []
        if node.type in node_types_of_interest.keys():
            # Only add as breakpoint if it's a top-level declaration
            if self.is_top_level_declaration(node):
                points_of_interest.append((node, node_types_of_interest[node.type]))

        for child in node.children:
            points_of_interest.extend(self.extract_points_of_interest(child, file_extension))

        return points_of_interest

    def _get_node_types_of_interest(self, file_extension: str) -> Dict[str, str]:
        node_types = {
            'py': {
                'import_statement': 'Import',
                'export_statement': 'Export',
                'class_definition': 'Class',
                'function_definition': 'Function',
            },
            'css': {
                'tag_name': 'Tag',
                '@media': 'Media Query',
            },
            'js': {
                'import_statement': 'Import',
                'export_statement': 'Export',
                'class_declaration': 'Class',
                'function_declaration': 'Function',
                'arrow_function': 'Arrow Function',
                'statement_block': 'Block',
            },
            'ts': {
                'import_statement': 'Import',
                'export_statement': 'Export',
                'class_declaration': 'Class',
                'function_declaration': 'Function',
                'arrow_function': 'Arrow Function',
                'statement_block': 'Block',
                'interface_declaration': 'Interface',
                'type_alias_declaration': 'Type Alias',
            },
            'php': {
                'namespace_definition': 'Namespace',
                'class_declaration': 'Class',
                'method_declaration': 'Method',
                'function_definition': 'Function',
                'interface_declaration': 'Interface',
                'trait_declaration': 'Trait',
            },
            'rb': {
                'class': 'Class',
                'method': 'Method',
                'module': 'Module',
                'singleton_class': 'Singleton Class',
                'begin': 'Begin Block',
            },
            'go': {
                'import_declaration': 'Import',
                'function_declaration': 'Function',
                'method_declaration': 'Method',
                'type_declaration': 'Type',
                'struct_type': 'Struct',
                'interface_type': 'Interface',
                'package_clause': 'Package'
            }
        }

        if file_extension in node_types.keys():
            return node_types[file_extension]
        elif file_extension == "jsx":
            return node_types["js"]
        elif file_extension == "tsx":
            return node_types["ts"]
        else:
            raise ValueError("Unsupported file type")
        

    def _get_nodes_for_comments(self, file_extension: str) -> Dict[str, str]:
        node_types = {
            'py': {
                'comment': 'Comment',
                'decorator': 'Decorator',  # Broadened category
            },
            'css': {
                'comment': 'Comment'
            },
            'js': {
                'comment': 'Comment',
                'decorator': 'Decorator',  # Broadened category
            },
            'ts': {
                'comment': 'Comment',
                'decorator': 'Decorator',
            },
            'php': {
                'comment': 'Comment',
                'attribute': 'Attribute',
            },
            'rb': {
                'comment': 'Comment',
            },
            'go': {
                'comment': 'Comment',
            }
        }

        if file_extension in node_types.keys():
            return node_types[file_extension]
        elif file_extension == "jsx":
            return node_types["js"]
        elif file_extension == "tsx":
            return node_types["ts"]
        else:
            raise ValueError("Unsupported file type")
        
    def extract_comments(self, node: Node, file_extension: str) -> List[Tuple[Node, str]]:
        node_types_of_interest = self._get_nodes_for_comments(file_extension)

        comments = []
        if node.type in node_types_of_interest:
            comments.append((node, node_types_of_interest[node.type]))

        for child in node.children:
            comments.extend(self.extract_comments(child, file_extension))

        return comments

    def get_lines_for_points_of_interest(self, code: str, file_extension: str) -> List[int]:
        language_name = self.language_extension_map.get(file_extension)
        if language_name is None:
            raise ValueError("Unsupported file type")

        language = self.languages.get(language_name)
        if language is None:
            raise ValueError("Language parser not found")

        parser = Parser()
        parser.set_language(language)

        tree = parser.parse(bytes(code, "utf8"))

        root_node = tree.root_node
        points_of_interest = self.extract_points_of_interest(root_node, file_extension)

        line_numbers_with_type_of_interest = {}

        for node, type_of_interest in points_of_interest:
            start_line = node.start_point[0] 
            if type_of_interest not in line_numbers_with_type_of_interest:
                line_numbers_with_type_of_interest[type_of_interest] = []

            if start_line not in line_numbers_with_type_of_interest[type_of_interest]:
                line_numbers_with_type_of_interest[type_of_interest].append(start_line)

        lines_of_interest = []
        for _, line_numbers in line_numbers_with_type_of_interest.items():
            lines_of_interest.extend(line_numbers)

        return lines_of_interest

    def get_lines_for_comments(self, code: str, file_extension: str) -> List[int]:
        language_name = self.language_extension_map.get(file_extension)
        if language_name is None:
            raise ValueError("Unsupported file type")

        language = self.languages.get(language_name)
        if language is None:
            raise ValueError("Language parser not found")

        parser = Parser()
        parser.set_language(language)

        tree = parser.parse(bytes(code, "utf8"))

        root_node = tree.root_node
        comments = self.extract_comments(root_node, file_extension)

        line_numbers_with_comments = {}

        for node, type_of_interest in comments:
            start_line = node.start_point[0] 
            if type_of_interest not in line_numbers_with_comments:
                line_numbers_with_comments[type_of_interest] = []

            if start_line not in line_numbers_with_comments[type_of_interest]:
                line_numbers_with_comments[type_of_interest].append(start_line)

        lines_of_interest = []
        for _, line_numbers in line_numbers_with_comments.items():
            lines_of_interest.extend(line_numbers)

        return lines_of_interest

    def print_all_line_types(self, code: str, file_extension: str):
        language_name = self.language_extension_map.get(file_extension)
        if language_name is None:
            print(f"Unsupported file type: {file_extension}")
            return

        language = self.languages.get(language_name)
        if language is None:
            print("Language parser not found")
            return

        parser = Parser()
        parser.set_language(language)
        tree = parser.parse(bytes(code, "utf8"))

        root_node = tree.root_node
        line_to_node_type = self.map_line_to_node_type(root_node)

        code_lines = code.split('\n')

        for line_num, node_types in line_to_node_type.items():
            line_content = code_lines[line_num - 1]  # Adjusting index for zero-based indexing
            print(f"line {line_num}: {', '.join(node_types)} | Code: {line_content}")


    def map_line_to_node_type(self, node, line_to_node_type=None, depth=0):
        if line_to_node_type is None:
            line_to_node_type = {}

        start_line = node.start_point[0] + 1  # Tree-sitter lines are 0-indexed; converting to 1-indexed

        # Only add the node type if it's the start line of the node
        if start_line not in line_to_node_type:
            line_to_node_type[start_line] = []
        line_to_node_type[start_line].append(node.type)

        for child in node.children:
            self.map_line_to_node_type(child, line_to_node_type, depth + 1)

        return line_to_node_type
    
