import re
import math
import datetime
import ast

# ─── AST SANDBOX ───────────────────────────────────────────────────────────────
# A string blocklist over code passed to exec() is cosmetic: the classic sandbox
# escape — ().__class__.__base__.__subclasses__()[...].__init__.__globals__[...] —
# uses no blocked keyword and works even with a restricted __builtins__, because
# attribute traversal does not depend on builtins. The real gate is structural:
# parse the code and reject any construct that could break out.

# Allowlisted AST node types. Anything not listed is rejected.
_ALLOWED_NODES = frozenset({
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.For, ast.While, ast.If, ast.Break, ast.Continue, ast.Pass,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.Call, ast.keyword, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.FormattedValue, ast.JoinedStr,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.Subscript, ast.Slice, ast.Starred,
    # Attribute access is allowed ONLY for non-underscore attrs (checked below);
    # this is what lets re.findall / text.split work while blocking __class__ etc.
    ast.Attribute,
    # operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
})


def _validate_ast_safe(code: str):
    """Returns None if the code is structurally safe, else a violation string.

    Forbids: imports, def/class/lambda, with/try/raise, global/nonlocal, and —
    crucially — any identifier or attribute beginning with '_' (closes dunder
    traversal). Only the helper functions and re/math/datetime remain reachable.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return f"SYNTAX_ERROR: {e}"

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            return f"SECURITY_VIOLATION: construct '{type(node).__name__}' is not allowed."
        # Block dunder / private traversal (the actual escape vector).
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return f"SECURITY_VIOLATION: attribute access to '{node.attr}' is not allowed."
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            return f"SECURITY_VIOLATION: identifier '{node.id}' is not allowed."
    return None


class LegalREPL:
    """
    A safe, read-only Python REPL for the Magistrate Agent to search facts in the text.
    Acts as the 'RLM' (Recursive Language Model) component.
    V2: Added search_money, search_dates_extended, search_parties.
    V2: Fixed security filter to use word boundaries.
    """
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.context = {
            "text": raw_text,
            "re": re,
            "math": math,
            "datetime": datetime,
            "search_dates": self.search_dates,
            "search_dates_extended": self.search_dates_extended,
            "search_money": self.search_money,
            "search_parties": self.search_parties,
            "grep": self.grep
        }

    def search_dates(self, keyword_context: str, window: int = 100):
        """
        Searches for dates (dd/mm/yyyy or dd-mm-yyyy) near a keyword.
        Example: search_dates("citação", 50) -> Finds dates within 50 chars of "citação"
        """
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

    def search_dates_extended(self, keyword_context: str = "", window: int = 200):
        """
        Searches for dates written in extended Portuguese format near a keyword.
        Captures: "17 de fevereiro de 2026", "1º de março de 2025", "fevereiro/2026"
        If keyword is empty, searches the entire text.
        """
        months = (
            r'(?:janeiro|fevereiro|mar[cç]o|abril|maio|junho|'
            r'julho|agosto|setembro|outubro|novembro|dezembro)'
        )
        # Patterns: "17 de fevereiro de 2026", "1º de março de 2025"
        extended_pattern = rf'\d{{1,2}}[º°]?\s+de\s+{months}\s+de\s+\d{{4}}'
        # Pattern: "fevereiro/2026" or "fevereiro de 2026"
        month_year_pattern = rf'{months}(?:\s+de\s+|/)\d{{4}}'

        combined = rf'(?:{extended_pattern}|{month_year_pattern})'
        
        if keyword_context:
            results = []
            for kw_match in re.finditer(re.escape(keyword_context), self.raw_text, re.IGNORECASE):
                start = max(0, kw_match.start() - window)
                end = min(len(self.raw_text), kw_match.end() + window)
                snippet = self.raw_text[start:end]
                
                dates = re.findall(combined, snippet, re.IGNORECASE)
                if dates:
                    results.append({"snippet": snippet, "dates": dates})
            return results if results else "NOT_FOUND"
        else:
            dates = re.findall(combined, self.raw_text, re.IGNORECASE)
            return dates if dates else "NOT_FOUND"

    def search_money(self, keyword_context: str = "", window: int = 200):
        """
        Searches for monetary values (R$ X.XXX,XX) near a keyword.
        Captures: R$ 1.000,00 / R$1000 / R$ 50.000 / 1.000,00
        If keyword is empty, finds all monetary values in the text.
        """
        # Monetary regex: R$ followed by digits with optional . and ,
        money_pattern = r'R\$\s*[\d]{1,3}(?:\.[\d]{3})*(?:,[\d]{2})?'
        
        if keyword_context:
            results = []
            for kw_match in re.finditer(keyword_context, self.raw_text, re.IGNORECASE):
                start = max(0, kw_match.start() - window)
                end = min(len(self.raw_text), kw_match.end() + window)
                snippet = self.raw_text[start:end]
                
                values = re.findall(money_pattern, snippet)
                if values:
                    results.append({"snippet": snippet, "values": values})
            return results if results else "NOT_FOUND"
        else:
            values = re.findall(money_pattern, self.raw_text)
            return values if values else "NOT_FOUND"

    def search_parties(self):
        """
        Searches for parties in the process (Autor/Requerente, Réu/Requerido).
        Returns a dict with author and defendant names.
        """
        results = {}
        
        # Common patterns for author
        author_patterns = [
            r'(?:autor|autora|requerente|demandante|exequente)[:\s]+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇa-záéíóúâêôãõç\s\.]+?)(?:\s*,|\s*\n|\s*CPF|\s*CNPJ|\s*inscrit|\s*qualificad|\s*brasileiro)',
            r'([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇa-záéíóúâêôãõç\s\.]+?)\s*(?:,\s*)?(?:autor|autora|requerente)',
        ]
        
        # Common patterns for defendant
        defendant_patterns = [
            r'(?:réu|ré|requerido|requerida|demandado|demandada|executado|executada)[:\s]+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇa-záéíóúâêôãõç\s\.]+?)(?:\s*,|\s*\n|\s*CPF|\s*CNPJ|\s*inscrit|\s*qualificad)',
            r'([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇa-záéíóúâêôãõç\s\.]+?)\s*(?:,\s*)?(?:réu|ré|requerido|requerida)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, self.raw_text, re.IGNORECASE)
            if match:
                results["autor"] = match.group(1).strip()
                break
        
        for pattern in defendant_patterns:
            match = re.search(pattern, self.raw_text, re.IGNORECASE)
            if match:
                results["reu"] = match.group(1).strip()
                break
        
        if not results:
            return "NOT_FOUND"
        
        return results

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
        Executes read-only Python in a structurally-restricted context.

        Security model (defense in depth):
          1. AST allowlist (`_validate_ast_safe`): rejects imports, def/class,
             and any '_'-prefixed name/attribute — closing dunder-traversal
             sandbox escapes that a string blocklist cannot.
          2. Restricted __builtins__: only pure, side-effect-free helpers.
          3. Restricted globals: only the legal-search helpers + re/math/datetime.
        """
        violation = _validate_ast_safe(code)
        if violation:
            return violation

        try:
            import io
            from contextlib import redirect_stdout

            _safe_builtins = {
                "len": len, "range": range, "enumerate": enumerate, "zip": zip,
                "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
                "str": str, "int": int, "float": float, "bool": bool,
                "sum": sum, "max": max, "min": min, "abs": abs, "round": round,
                "print": print, "map": map, "filter": filter, "any": any, "all": all,
                "isinstance": isinstance,
            }
            # self.context already exposes text + the safe helpers; inject only
            # the restricted builtins. No __import__ is provided.
            exec_globals = {"__builtins__": _safe_builtins}
            f = io.StringIO()
            with redirect_stdout(f):
                exec(compile(code, "<legal_repl>", "exec"), exec_globals, self.context)

            output = f.getvalue()
            return output if output else "CODE_EXECUTED_NO_OUTPUT"

        except Exception as e:
            return f"RUNTIME_ERROR: {str(e)}"
