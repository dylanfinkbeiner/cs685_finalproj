# Lengths (in tokens) of arguments in test set
    all_one_X = []
    for s in X.values():
        all_one_X.extend(s)
    plt.hist([len(x['arg_tokens']) for x in all_one_X])

