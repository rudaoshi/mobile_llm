import tensorflow as tf
import tensorflow_datasets as tfds
import numpy
nqa = tfds.load('natural_questions/longt5', as_supervised=False)
print(nqa['train'])

prefetchdataset = nqa['train']
print(len(prefetchdataset))

def remove_first_character(string):
  return string[2:-1]

all_qna = []
n=0
samples = 307373
for element in prefetchdataset:
  if  "NULL"in str(element['answer'].numpy()):
    continue
  tensordata = element['question']+" = " + element['answer']
  stringdata = remove_first_character(str(tensordata.numpy()))
  all_qna.append(stringdata)

  n+=1
  if n==samples:
   break
print(all_qna)

# Name of the text file
file_name = "output.txt"

# Open the file in write mode and write each string to a new line
with open(file_name, 'w') as file:
    for string in all_qna:
        file.write(f"{string}\n")

print(f"Strings written to {file_name}")