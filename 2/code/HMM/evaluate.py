gold_txt = []
with open("./CTB_test_gold.utf8", encoding="utf-8") as fg:
    for gold_sentence in fg.readlines():
        gold_txt.append(gold_sentence.split())
result_txt = []
with open("./result.txt", encoding="utf-8") as fr:
    for result_sentence in fr.readlines():
        result_txt.append(result_sentence.split())

gold_word_number = 0
result_word_number = 0
success_word_number = 0

for i in range(len(gold_txt)):
    gold_sentence = gold_txt[i]
    result_sentence = result_txt[i]
    gold_word_number += len(gold_sentence)
    result_word_number += len(result_sentence)
    for word in result_sentence:
        if word in gold_sentence:
            success_word_number += 1

precision = success_word_number / result_word_number
recall = success_word_number / gold_word_number
f1 = 2*precision*recall / (precision+recall)

print("precision: " + str(precision))
print("recall: " + str(recall))
print("f1: " + str(f1))
