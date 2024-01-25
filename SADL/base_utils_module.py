
def check(value, class_type, min, max, choices):
    # param should be type class_type
    if not isinstance(value, (class_type)):
        print('{valuee} is not set to {class_typee}'.format(
            valuee=value, class_typee=class_type))
        return False
    
    if min != None and max != None: 
        if value < min and value >= max:
            print('not between minmax')
            return False
    return True
        
    