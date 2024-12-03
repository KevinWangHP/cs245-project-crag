import re
str1 = "    1  2 2             4 \n 4   45  3333"
str1 = re.sub(r'\n+', '  ', re.sub(r' +', ' ', str1))
print(str1)
for i in str1:
    print(i)