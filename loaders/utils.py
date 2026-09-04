import ast

class DotDict(dict):
    """A dictionary that supports dot notation and nested dot access, and converts numeric and tuple-like strings to native types."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        del self[key]
    
    def update(self, *args, **kwargs):
        """Override update to recursively convert items."""
        for k, v in dict(*args, **kwargs).items():
            self[k] = self._convert(v)
    
    def _convert(self, value):
        if isinstance(value, dict):
            return DotDict(value)
        elif isinstance(value, list):
            return [self._convert(i) for i in value]
        elif isinstance(value, str):
            # Check for None string
            if value == "None":
                return None
            
            # Check for boolean strings
            if value == "True":
                return True
            if value == "False":
                return False
            
            # Try to parse tuples
            if value.startswith("(") and value.endswith(")"):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, tuple):
                        # Recursively convert tuple items
                        return tuple(self._convert(i) for i in parsed)
                except Exception:
                    pass
            
            # Try to parse numbers (int or float)
            try:
                if '.' in value or 'e' in value.lower():
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                # Not a number, return as-is
                return value
        else:
            return value