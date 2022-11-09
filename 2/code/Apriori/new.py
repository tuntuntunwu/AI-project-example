data_set = []
with open("./Groceries.txt", encoding='UTF-8') as f:
    for transcation in f.readlines():
        transcation = transcation[1:-2]
        trlist = transcation.split(',')
        data_set.append(trlist)
    print(data_set)