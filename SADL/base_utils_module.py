
def check(name, value, class_type, min, max, choices):
    print(value)
    print(class_type)
    print(min)
    print(max)
    print(choices)
    # param should be type class_type
    if not isinstance(value, (class_type)):
        print(f"[{name} = {value}] type is not {class_type}")
        return False
    
    if min != None and max != None: 
        if not (value >= min and value <= max):
            print(f"[{name} = {value}] not between min = {min} and max = {max} values")
            return False
    return True
        
    