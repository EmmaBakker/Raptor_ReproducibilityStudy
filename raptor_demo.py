import os
from raptor import RetrievalAugmentation

RA = RetrievalAugmentation()

text = "Cinderella was kind and brave. With help, she went to the ball and found her happy ending."
RA.add_documents(text)

question = "How did Cinderella reach her happy ending?"
answer = RA.answer_question(question=question)
print("Answer:", answer)