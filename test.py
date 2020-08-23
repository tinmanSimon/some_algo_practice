#mode == 1 means we want to print both produced result and expected result.
def test(func, data, cmp_func, mode = 0):
    for d in data:
        ret = func(*d[0])
        cmp_result = cmp_func(ret, d[1])
        print("Test data:", d[0], ", Test result:", cmp_result)
        if cmp_result == False or mode == 1: print("Produced result:", ret, "\nExpected result:", d[1])
        if mode == 1: print("\n")
