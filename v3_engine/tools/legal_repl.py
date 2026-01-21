import re
import math
import datetime

class LegalREPL:
    """
    A safe, read-only Python REPL for the Magistrate Agent to search facts in the text.
    Acts as the 'RLM' (Recursive Language Model) component.
    """
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.context = {
            "text": raw_text,
            "re": re,
            "math": math,
            "datetime": datetime,
            "search_dates": self.search_dates,
            "grep": self.grep
        }

    def search_dates(self, keyword_context: str, window: int = 100):
        """
        Searches for dates near a keyword.
        Example: search_dates("citação", 50) -> Finds dates within 50 chars of "citação"
        """
        # Finds keyword (case insensitive)
        matches = []
        for match in re.finditer(re.escape(keyword_context), self.raw_text, re.IGNORECASE):
            start = max(0, match.start() - window)
            end = min(len(self.raw_text), match.end() + window)
            snippet = self.raw_text[start:end]
            
            # Simple Date Regex (dd/mm/yyyy or dd-mm-yyyy)
            dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', snippet)
            if dates:
                matches.append({"snippet": snippet, "dates": dates})
        
        return matches if matches else "NOT_FOUND"

    def grep(self, pattern: str, context_lines: int = 1):
        """
        Mimics grep. Returns lines matching the regex pattern.
        """
        try:
            lines = self.raw_text.split('\n')
            results = []
            compiled_re = re.compile(pattern, re.IGNORECASE)
            
            for i, line in enumerate(lines):
                if compiled_re.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    results.append("\n".join(lines[start:end]))
            
            return results if results else "NOT_FOUND"
        except Exception as e:
            return f"REGEX_ERROR: {str(e)}"

    def run_code(self, code: str):
        """
        Executes the provided Python code in the safe context.
        """
        # Security: Block dangerous imports
        if "os" in code or "sys" in code or "subprocess" in code or "open(" in code:
            return "SECURITY_VIOLATION: File system access prohibited."
            
        try:
            # Capture std out? No, just eval or exec.
            # We want the agent to use 'print' or just expression
            # Use a buffer to capture print output
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                exec(code, self.context)
            
            output = f.getvalue()
            return output if output else "CODE_EXECUTED_NO_OUTPUT"
            
        except Exception as e:
            return f"RUNTIME_ERROR: {str(e)}"
