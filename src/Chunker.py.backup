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
        code_parser = CodeParser(self.file_extension)
        chunks = {}
        current_chunk = ""
        token_count = 0
        lines = code.split("\n")
        i = 0
        chunk_number = 1
        start_line = 0
        breakpoints = sorted(code_parser.get_lines_for_points_of_interest(code, self.file_extension))
        comments = sorted(code_parser.get_lines_for_comments(code, self.file_extension))
        adjusted_breakpoints = []
        for bp in breakpoints:
            current_line = bp - 1
            highest_comment_line = None  # Initialize with None to indicate no comment line has been found yet
            while current_line in comments:
                highest_comment_line = current_line  # Update highest comment line found
                current_line -= 1  # Move to the previous line

            if highest_comment_line:  # If a highest comment line exists, add it
                adjusted_breakpoints.append(highest_comment_line)
            else:
                adjusted_breakpoints.append(
                    bp)  # If no comments were found before the breakpoint, add the original breakpoint

        breakpoints = sorted(set(adjusted_breakpoints))  # Ensure breakpoints are unique and sorted

        while i < len(lines):
            line = lines[i]
            new_token_count = count_tokens(line, self.encoding_name)
            if token_count + new_token_count > token_limit:

                # Set the stop line to the last breakpoint before the current line
                if i in breakpoints:
                    stop_line = i
                else:
                    stop_line = max(max([x for x in breakpoints if x < i], default=start_line), start_line)

                # If the stop line is the same as the start line, it means we haven't reached a breakpoint yet and we need to move to the next line to find one
                if stop_line == start_line and i not in breakpoints:
                    token_count += new_token_count
                    i += 1

                # If the stop line is the same as the start line and the current line is a breakpoint, it means we can create a chunk with just the current line
                elif stop_line == start_line and i == stop_line:
                    token_count += new_token_count
                    i += 1


                # If the stop line is the same as the start line and the current line is a breakpoint, it means we can create a chunk with just the current line
                elif stop_line == start_line and i in breakpoints:
                    current_chunk = "\n".join(lines[start_line:stop_line])
                    if current_chunk.strip():  # If the current chunk is not just whitespace
                        # Lines are 0-based internally; report 1-based inclusive
                        chunks[chunk_number] = {
                            "text": current_chunk,
                            "start_line": start_line + 1,
                            "end_line": max(stop_line, start_line)  # stop_line is exclusive; equals start_line here
                        }
                        chunk_number += 1

                    token_count = 0
                    start_line = i
                    i += 1

                # If the stop line is different from the start line, it means we're at the end of a block
                else:
                    current_chunk = "\n".join(lines[start_line:stop_line])
                    if current_chunk.strip():
                        chunks[chunk_number] = {
                            "text": current_chunk,
                            "start_line": start_line + 1,
                            "end_line": max(stop_line, start_line)  # stop_line is exclusive
                        }
                        chunk_number += 1

                    i = stop_line
                    token_count = 0
                    start_line = stop_line
            else:
                # If the token count is still within the limit, add the line to the current chunk
                token_count += new_token_count
                i += 1

        # Append remaining code, if any, ensuring it's not empty or whitespace
        current_chunk_code = "\n".join(lines[start_line:])
        if current_chunk_code.strip():  # Checks if the chunk is not just whitespace
            chunks[chunk_number] = {
                "text": current_chunk_code,
                "start_line": start_line + 1,
                "end_line": len(lines)
            }

        return chunks

    def get_chunk(self, chunked_codebase, chunk_number):
        return chunked_codebase[chunk_number]
