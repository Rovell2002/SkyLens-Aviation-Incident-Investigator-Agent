import retriever as R

for h in R.search("landing gear failed to extend", n=5):
    print(h["acn"], h["date"], h["aircraft"], "|", h["text"][:200], "\n")
