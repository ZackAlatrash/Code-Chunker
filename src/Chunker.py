from abc import ABC, abstractmethod
from CodeParser import CodeParser
from utils import count_tokens


class Chunker(ABC):
    def __init__(self, encoding_name="gpt-4"):
        self.encoding_name = encoding_name

    @abstractmethod
    def chunk(self, content, token_limit):
        pass

    @abstractmethod
    def get_chunk(self, chunked_content, chunk_number):
        pass

    @staticmethod
    def print_chunks(chunks):
        for chunk_number, chunk_val in chunks.items():
            # Support both legacy string chunks and new dict chunks
            if isinstance(chunk_val, dict):
                header = f"Chunk {chunk_number} (lines {chunk_val.get('start_line')}–{chunk_val.get('end_line')}):"
                text = chunk_val.get("text", "")
            else:
                header = f"Chunk {chunk_number}:"
                text = str(chunk_val)

            print(header)
            print("=" * 40)
            print(text)
            print("=" * 40)

    @staticmethod
    def consolidate_chunks_into_file(chunks):
        # Handle dict-valued chunks (new) and string-valued (legacy)
        parts = []
        for v in chunks.values():
            parts.append(v["text"] if isinstance(v, dict) else str(v))
        return "\n".join(parts)

    @staticmethod
    def count_lines(consolidated_chunks):
        lines = consolidated_chunks.split("\n")
        return len(lines)


class CodeChunker(Chunker):
    def __init__(self, file_extension, encoding_name="gpt-4"):
        super().__init__(encoding_name)
        self.file_extension = file_extension

    def chunk(self, code, token_limit) -> dict:
        """
        Chunk code with STRICT one-function-per-chunk policy.
        
        Strategy:
        1. Create a chunk boundary at EVERY function/class breakpoint
        2. Allow single functions to exceed token_limit (acceptable)
        3. Don't merge adjacent small functions
        
        This ensures clean semantic boundaries where each chunk typically
        contains one complete function/method/class.
        """
        code_parser = CodeParser(self.file_extension)
        chunks = {}
        lines = code.split("\n")
        chunk_number = 1
        
        # Get breakpoints (function/class starts) from tree-sitter
        breakpoints = sorted(code_parser.get_lines_for_points_of_interest(code, self.file_extension))
        comments = sorted(code_parser.get_lines_for_comments(code, self.file_extension))
        
        # Adjust breakpoints to include preceding comments
        adjusted_breakpoints = []
        for bp in breakpoints:
            current_line = bp - 1
            highest_comment_line = None
            while current_line in comments:
                highest_comment_line = current_line
                current_line -= 1
            
            if highest_comment_line:
                adjusted_breakpoints.append(highest_comment_line)
            else:
                adjusted_breakpoints.append(bp)
        
        breakpoints = sorted(set(adjusted_breakpoints))
        
        # Add end-of-file as final breakpoint
        if not breakpoints or breakpoints[-1] < len(lines):
            breakpoints.append(len(lines))
        
        # Create chunks at every breakpoint (strict policy)
        for idx in range(len(breakpoints)):
            start_line = breakpoints[idx] if idx == 0 else breakpoints[idx]
            
            # Determine end line
            if idx + 1 < len(breakpoints):
                end_line = breakpoints[idx + 1]
            else:
                end_line = len(lines)
            
            # Extract chunk text
            if idx == 0 and start_line > 0:
                # Handle file header (imports, package declaration, etc.)
                chunk_text = "\n".join(lines[0:start_line])
                if chunk_text.strip():
                    chunks[chunk_number] = {
                        "text": chunk_text,
                        "start_line": 1,
                        "end_line": start_line
                    }
                    chunk_number += 1
            
            # Create chunk for this function/class
            chunk_text = "\n".join(lines[start_line:end_line])
            if chunk_text.strip():
                chunks[chunk_number] = {
                    "text": chunk_text,
                    "start_line": start_line + 1,  # Convert to 1-based
                    "end_line": end_line  # Already exclusive, so end_line is correct
                }
                chunk_number += 1
        
        # If no breakpoints found (e.g., file with just variables), return entire file
        if not chunks:
            chunks[1] = {
                "text": code,
                "start_line": 1,
                "end_line": len(lines)
            }
        
        return chunks

    def get_chunk(self, chunked_codebase, chunk_number):
        return chunked_codebase[chunk_number]
