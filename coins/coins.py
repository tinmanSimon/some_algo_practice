#returns a list of all possible combinations of coins to represent value of n cents.
def coins(n, coins_values = [25, 10, 5, 1]):
    dp = {}

    def coins_helper(k, highest_change):
        if k == 0: 
            return [""]
        if k in dp and highest_change in dp[k]:
            if k >= 5 and k % 5 == 0: 
                t = k
            return dp[k][highest_change]
        res = []
        for coin_val in coins_values:
            if k >= coin_val and highest_change >= coin_val:
                ret = coins_helper(k - coin_val, coin_val)
                for coinstr in ret:
                    res.append(str(coin_val) + " " + coinstr)
        if not k in dp: dp[k] = {}
        dp[k][highest_change] = res
        return res
    result = coins_helper(n, coins_values[0])
    return result

print(coins(50, [25, 10, 7, 3]))
