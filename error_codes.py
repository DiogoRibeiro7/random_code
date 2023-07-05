class MissingColumnError(Exception):
    def __init__(self, column_name):
        self.column_name = column_name
        self.message = f"Missing column: {column_name}"
        super().__init__(self.message)

class InvalidValueError(Exception):
    def __init__(self, column_name, invalid_value):
        self.column_name = column_name
        self.invalid_value = invalid_value
        self.message = f"Invalid value '{invalid_value}' in column '{column_name}'"
        super().__init__(self.message)

class InvalidDataTypeError(Exception):
    def __init__(self, column_name, expected_type, actual_type):
        self.column_name = column_name
        self.expected_type = expected_type
        self.actual_type = actual_type
        self.message = f"Invalid data type in column '{column_name}'. Expected type: {expected_type}, Actual type: {actual_type}"
        super().__init__(self.message)
        
        
class CustomAttributeError(AttributeError):
    def __init__(self, object_name, attribute_name):
        self.object_name = object_name
        self.attribute_name = attribute_name
        self.message = f"Attribute '{attribute_name}' does not exist for object '{object_name}'"
        super().__init__(self.message)










try:
    # Code that may raise custom errors or general exceptions
    # ...

    # Example code that may raise custom errors
    column_name = 'Salary'
    if column_name not in df.columns:
        raise CustomError1(f"Missing column: {column_name}")
    elif df[column_name].dtype != int:
        raise CustomError2(f"Invalid data type in column '{column_name}'. Expected type: int, Actual type: {df[column_name].dtype}")

    # Other code statements...
    # ...

except CustomError1 as e:
    # Handle CustomError1
    print(f"Custom Error 1: {e.message}")

except CustomError2 as e:
    # Handle CustomError2
    print(f"Custom Error 2: {e.message}")

except Exception as e:
    # Handle other general exceptions or re-raise if not a custom error
    if isinstance(e, CustomError):
        # Log or handle the custom error here
        print("Custom Error occurred")
    else:
        # Re-raise the general exception
        raise e