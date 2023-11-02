class BothFalseError(Exception):
    """Exception raised when both boolean values are False"""
    def __init__(self):
        super().__init__("Both values cannot be False")
    
    @staticmethod
    def check_booleans(bool1, bool2):
        if not bool1 and not bool2:
            raise BothFalseError()
        
if __name__ == '__main__':
    pass