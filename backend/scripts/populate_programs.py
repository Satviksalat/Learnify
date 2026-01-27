import json
import os

# Define solutions map: ID -> Code
solutions = {
    1: "print('Hello World. This is Python.')",
    2: "a = 10\nb = 10.5\nc = 3 + 4j\nd = True\ne = 'Hello'\n\nprint(type(a))\nprint(type(b))\nprint(type(c))\nprint(type(d))\nprint(type(e))",
    3: "x = 10\nprint('ID:', id(x))\nprint('Type:', type(x))\n\nfor i in range(5):\n    print(i, end=' ')",
    4: "# Implicit\na = 10\nb = 2.5\nc = a + b\nprint(c, type(c))\n\n# Explicit\nx = 10\ny = float(x)\nprint(y, type(y))\nz = str(x)\nprint(z, type(z))",
    5: "a = 10\nb = 5\n\n# Arithmetic\nprint('Add:', a + b)\n\n# Relational\nprint('Greater:', a > b)\n\n# Assignment\nc = a\nc += b\nprint('Assign:', c)\n\n# Logical\nprint('And:', a > 5 and b < 10)\n\n# Bitwise\nprint('Bitwise Or:', a | b)\n\n# Ternary\nres = 'High' if a > 8 else 'Low'\nprint(res)",
    6: "name = input('Enter Name: ')\nprint('Hello', name, sep='-', end='!')\nprint('\\nWelcome to {} tutorial'.format('Python'))",
    7: "num = 10\nif num > 0:\n    print('Positive')\nelif num < 0:\n    print('Negative')\nelse:\n    print('Zero')",
    8: "# While loop\ni = 1\nwhile i <= 5:\n    print(i, end=' ')\n    i += 1\n\nprint()\n\n# For loop\nfor j in range(1, 6):\n    print(j, end=' ')",
    9: "for i in range(10):\n    if i == 3:\n        continue\n    if i == 8:\n        break\n    if i == 5:\n        pass\n    print(i, end=' ')",
    10: "s = 'Python Programming'\nprint(s[0])       # P\nprint(s[-1])      # g\nprint(s[0:6])     # Python\nprint(s[::2])     # Pto rgamn\nprint(s[::-1])    # gnimmargorP nohtyP",
    11: "# Write\nwith open('test.txt', 'w') as f:\n    f.write('Hello Python')\n\n# Read\nwith open('test.txt', 'r') as f:\n    print(f.read())",
    12: "source = 'test.txt'\ndest = 'copy.txt'\n\nwith open(source, 'r') as f1, open(dest, 'w') as f2:\n    for line in f1:\n        f2.write(line)",
    13: "filename = 'test.txt'\nfreq = {}\n\nwith open(filename, 'r') as f:\n    text = f.read()\n    for chat in text:\n        if chat in freq:\n            freq[chat] += 1\n        else:\n            freq[chat] = 1\nprint(freq)",
    14: "filename = 'test.txt'\nwith open(filename, 'r') as f:\n    lines = f.readlines()\n    for line in lines[::-1]:\n        print(line.strip())",
    15: "filename = 'test.txt'\nchars = words = lines = 0\n\nwith open(filename, 'r') as f:\n    for line in f:\n        lines += 1\n        words += len(line.split())\n        chars += len(line)\n\nprint('Lines:', lines)\nprint('Words:', words)\nprint('Chars:', chars)",
    16: "l1 = [1, 2, 3]\nl2 = list((4, 5, 6))\nl3 = [x for x in range(5)]\nprint(l1, l2, l3)",
    17: "lst = [3, 1, 4, 1, 5]\nprint(len(lst))\nlst.append(9)\nlst.sort()\nprint(lst)\nlst.pop()\nprint(lst)",
    18: "t1 = (1, 2, 3)\nt2 = tuple((4, 5, 6))\nt3 = 10,  # Singleton\nprint(t1, t2, t3)",
    19: "rows = 5\nfor i in range(1, rows + 1):\n    print('* ' * i)\n\nprint()\n\nfor i in range(1, rows + 1):\n    for j in range(1, i + 1):\n        print(j, end=' ')\n    print()",
    20: "t = (5, 2, 9, 1, 5)\nprint('Count 5:', t.count(5))\nprint('Index 2:', t.index(2))\nprint('Sorted:', sorted(t))\nprint('Max:', max(t))",
    21: "s1 = {1, 2, 3}\ns2 = set([3, 4, 5])\nprint(s1, s2)",
    22: "A = {1, 2, 3}\nB = {3, 4, 5}\n\nprint('Union:', A | B)\nprint('Intersection:', A & B)\nprint('Difference:', A - B)\nA.add(6)\nprint(A)",
    23: "d1 = {'a': 1, 'b': 2}\nd2 = dict(name='John', age=30)\nprint(d1, d2)",
    24: "d = {'a': 1, 'b': 2}\nprint(d.get('a'))\nprint(d.keys())\nd.update({'c': 3})\nprint(d)",
    25: "def get_stats(nums):\n    return min(nums), max(nums), sum(nums)\n\nmn, mx, sm = get_stats([1, 2, 3, 4, 5])\nprint(mn, mx, sm)",
    26: "g = 10 \n\ndef func():\n    global g\n    l = 5\n    g += 1\n    print('Local:', l, 'Global:', g)\n\nfunc()",
    27: "sq = lambda x: x * x\nadd = lambda a, b: a + b\n\nprint(sq(5))\nprint(add(3, 4))",
    28: "import datetime\nname = input('Name: ')\nage = int(input('Age: '))\ncurrent_year = datetime.datetime.now().year\nturn_60 = current_year + (60 - age)\nprint(f'{name} will turn 60 in {turn_60}')",
    29: "num = int(input('Enter number: '))\nif num % 2 == 0:\n    print('Even')\nelse:\n    print('Odd')",
    30: "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a, end=' ')\n        a, b = b, a + b\n\nfib(10)",
    31: "def reverse_val(val):\n    return str(val)[::-1]\n\nprint(reverse_val('Python'))\nprint(reverse_val(12345))",
    32: "def is_armstrong(num):\n    s = 0\n    temp = num\n    while temp > 0:\n        digit = temp % 10\n        s += digit ** 3\n        temp //= 10\n    return num == s\n\nprint(is_armstrong(153))",
    33: "def is_palindrome(s):\n    return str(s) == str(s)[::-1]\n\nprint(is_palindrome('radar'))\nprint(is_palindrome('hello'))",
    34: "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))",
    35: "def is_vowel(char):\n    return char.lower() in 'aeiou'\n\nprint(is_vowel('a'))\nprint(is_vowel('z'))",
    36: "def custom_len(iterable):\n    count = 0\n    for _ in iterable:\n        count += 1\n    return count\n\nprint(custom_len('hello'))\nprint(custom_len([1, 2, 3]))",
    37: "lst = ['a', 'b', 'c', 'd', 'e', 'f']\nindices = [0, 2, 3, 5]\nres = [x for i, x in enumerate(lst) if i not in indices]\nprint(res)",
    38: "d = {'a': 3, 'b': 1, 'c': 2}\nsorted_d = dict(sorted(d.items(), key=lambda item: item[1]))\nprint(sorted_d)",
    39: "d = {'a': 100, 'b': 200, 'c': 300}\nprint(sum(d.values()))",
    40: "# Bubble Sort Example\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n\nprint(bubble_sort([64, 34, 25, 12, 22]))",
    41: "class Employee:\n    def __init__(self, name, salary):\n        self.name = name\n        self.__salary = salary # Private\n    \n    def get_salary(self):\n        return self.__salary\n\nemp = Employee('John', 5000)\nprint(emp.name)\nprint(emp.get_salary())",
    42: "import matplotlib.pyplot as plt\n\nx = [1, 2, 3, 4]\ny = [10, 20, 25, 30]\n\nplt.plot(x, y)\nplt.title('Line Plot')\n# plt.show()",
    43: "import matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 10, 100)\ny = np.sin(x)\n\nplt.plot(x, y)\n# plt.show()",
    44: "def knapsack(val, wt, W, n):\n    if n == 0 or W == 0:\n        return 0\n    if (wt[n-1] > W):\n        return knapsack(val, wt, W, n-1)\n    else:\n        return max(val[n-1] + knapsack(val, wt, W-wt[n-1], n-1),\n                   knapsack(val, wt, W, n-1))\n\nprint(knapsack([60, 100, 120], [10, 20, 30], 50, 3))",
    45: "def merge_sort(arr):\n    if len(arr) > 1:\n        mid = len(arr)//2\n        L = arr[:mid]\n        R = arr[mid:]\n        merge_sort(L)\n        merge_sort(R)\n        i = j = k = 0\n        while i < len(L) and j < len(R):\n            if L[i] < R[j]: arr[k] = L[i]; i += 1\n            else: arr[k] = R[j]; j += 1\n            k += 1\n        while i < len(L): arr[k] = L[i]; i += 1; k += 1\n        while j < len(R): arr[k] = R[j]; j += 1; k += 1\n\narr = [12, 11, 13, 5, 6, 7]\nmerge_sort(arr)\nprint(arr)",
    46: "import socket\ns = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\nprint('Socket Created')",
    47: "import socket\nhostname = socket.gethostname()\nip = socket.gethostbyname(hostname)\nprint('IP:', ip)",
    48: "import requests\nurl = 'https://www.example.com'\nr = requests.get(url)\nprint(r.text[:500])",
    49: "import urllib.request\nurl = 'https://www.python.org'\nurllib.request.urlretrieve(url, 'python_page.html')\nprint('Downloaded')",
    50: "import requests\nurl = 'https://www.python.org/static/img/python-logo.png'\nr = requests.get(url)\nwith open('logo.png', 'wb') as f:\n    f.write(r.content)\nprint('Image Saved')",
    51: "# Server\n# import socket\n# s = socket.socket()\n# s.bind(('localhost', 12345))\n# s.listen(5)\n# c, addr = s.accept()\n# print('Got connection from', addr)\n# c.send(b'Thank you for connecting')\n# c.close()",
    52: "# Client\n# import socket\n# s = socket.socket()\n# s.connect(('localhost', 12345))\n# print(s.recv(1024))\n# s.close()",
    53: "# UDP Server\n# import socket\n# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n# s.bind(('localhost', 12345))\n# while True:\n#     data, addr = s.recvfrom(1024)\n#     print('Message:', data)\n",
    54: "# File Server Logic\n# Open file, read bytes, s.send(bytes), close",
    55: "import smtplib\n# server = smtplib.SMTP('smtp.gmail.com', 587)\n# server.starttls()\n# server.login('email@gmail.com', 'password')\n# server.sendmail('from@gmail.com', 'to@gmail.com', 'Msg')\n# server.quit()",
    56: "import tkinter as tk\nroot = tk.Tk()\nl = tk.Label(root, text='Hello')\nl.pack()\ne = tk.Entry(root)\ne.pack()\nb = tk.Button(root, text='Click', command=lambda: print(e.get()))\nb.pack()\n# root.mainloop()",
    57: "import tkinter.messagebox\n#  tkinter.messagebox.showinfo('Title', 'Message')",
    58: "import tkinter as tk\nroot = tk.Tk()\ndef calc():\n    res = eval(entry.get())\n    label.config(text=str(res))\nentry = tk.Entry(root)\nentry.pack()\ntk.Button(root, text='Calculate', command=calc).pack()\nlabel = tk.Label(root)\nlabel.pack()\n# root.mainloop()",
    59: "import mysql.connector\n# mydb = mysql.connector.connect(host='localhost', user='root', password='')\n# mycursor = mydb.cursor()\n# mycursor.execute('SHOW DATABASES')\n# for x in mycursor: print(x)",
    60: "# Insert\n# sql = 'INSERT INTO customers (name, address) VALUES (%s, %s)'\n# val = ('John', 'Highway 21')\n# mycursor.execute(sql, val)\n# mydb.commit()",
    61: "# Update\n# sql = 'UPDATE customers SET address = %s WHERE address = %s'\n# val = ('Valley 345', 'Highway 21')\n# mycursor.execute(sql, val)\n# mydb.commit()",
    62: "# Delete\n# sql = 'DELETE FROM customers WHERE address = %s'\n# val = ('Mountain 21',)\n# mycursor.execute(sql, val)\n# mydb.commit()",
    63: "# Create DB script\n# mycursor.execute('CREATE DATABASE dbStudent')\n# mycursor.execute('CREATE TABLE tblStudInfo (id INT, name VARCHAR(255))')",
    64: "# Update Student\n# sql = 'UPDATE tblStudInfo SET name = \"Jane\" WHERE id = 1'",
    65: "# Delete Student\n# sql = 'DELETE FROM tblStudInfo WHERE id = 1'",

    # Machine Learning (101-151)
    101: "import pandas as pd\ndata = [['Alex', 10], ['Bob', 12], ['Clarke', 13]]\ndf = pd.DataFrame(data, columns=['Name', 'Age'])\nprint(df)",
    102: "import pandas as pd\ndf = pd.DataFrame({'A': [1, 2, 3]})\nres = []\nfor x in df['A']:\n    res.append(x * 2)\ndf['B'] = res\nprint(df)",
    103: "import pandas as pd\ndf = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})\ndf = df.rename(columns={'A': 'Col1', 'B': 'Col2'})\ndf = df.rename(index={0: 'row1', 1: 'row2'})\nprint(df)",
    104: "from sklearn import datasets\niris = datasets.load_iris()\nX, y = iris.data, iris.target\nprint(X.shape)",
    105: "import pandas as pd\ndf = pd.DataFrame({'A':[1,2,3], 'B':[4,5,6]})\n# Extract row 0, col 'A'\nprint(df.loc[0, 'A'])\n# Slice\nprint(df.iloc[0:2, :])",
    106: "from sklearn.impute import SimpleImputer\nimport numpy as np\nimp = SimpleImputer(missing_values=np.nan, strategy='mean')\ndata = [[1, 2], [np.nan, 3], [7, 6]]\nprint(imp.fit_transform(data))",
    107: "from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\ndata = ['cat', 'dog', 'cat', 'bird']\nprint(le.fit_transform(data))",
    108: "from sklearn.preprocessing import OneHotEncoder\nenc = OneHotEncoder()\ndata = [['male'], ['female'], ['male']]\nprint(enc.fit_transform(data).toarray())",
    109: "from sklearn.model_selection import train_test_split\nimport numpy as np\nX, y = np.arange(10).reshape((5, 2)), range(5)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)\nprint(X_train)",
    110: "from sklearn.preprocessing import StandardScaler\ndata = [[0, 0], [0, 0], [1, 1], [1, 1]]\nscaler = StandardScaler()\nprint(scaler.fit_transform(data))",
    111: "from sklearn.preprocessing import Normalizer\ndata = [[4, 1, 2, 2], [1, 3, 9, 3]]\ntrans = Normalizer().fit(data)\nprint(trans.transform(data))",
    112: "import numpy as np\nmat = np.matrix([[1, 2], [3, 4]])\nprint(mat.T) # Transpose",
    113: "from sklearn.preprocessing import scale\nimport numpy as np\ndata = np.array([[1, 2, 3], [4, 5, 6]])\nprint(scale(data))",
    114: "from sklearn.preprocessing import MinMaxScaler\ndata = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]\nscaler = MinMaxScaler()\nprint(scaler.fit_transform(data))",
    115: "from sklearn.preprocessing import Binarizer\nX = [[1, -1, 2], [2, 0, 0], [0, 1, -1]]\ntrans = Binarizer(threshold=0.0)\nprint(trans.fit_transform(X))",
    116: "from sklearn.linear_model import LinearRegression\nX = [[1], [2], [3]]\ny = [1, 2, 3]\nreg = LinearRegression().fit(X, y)\nprint(reg.coef_)",
    117: "from sklearn.metrics import mean_squared_error, r2_score\ny_true = [3, -0.5, 2, 7]\ny_pred = [2.5, 0.0, 2, 8]\nprint(mean_squared_error(y_true, y_pred))",
    118: "# Conceptual Example\n# df = pd.read_csv('advertising.csv')\n# reg.fit(df[['TV']], df['Sales'])",
    119: "import pandas as pd\ndf = pd.DataFrame({'A': [1, None, 3]})\nprint(df.isnull().sum())\ndf = df.fillna(0)\nprint(df)",
    120: "import matplotlib.pyplot as plt\nplt.scatter([1,2,3], [4,5,6])\n# plt.show()",
    121: "import seaborn as sns\nimport pandas as pd\nimport matplotlib.pyplot as plt\ndf = pd.DataFrame({'A':[1,2,3], 'B':[3,2,1]})\nsns.heatmap(df.corr(), annot=True)\n# plt.show()",
    122: "import pandas as pd\ndf = pd.DataFrame({'A': [1, 2, 3]})\nprint(df.describe())",
    123: "from sklearn.linear_model import LogisticRegression\nX = [[0], [1]]\ny = [0, 1]\nclf = LogisticRegression().fit(X, y)\nprint(clf.predict([[0.5]]))",
    124: "# Normal distribution sampling\nimport numpy as np\ns = np.random.normal(0, 0.1, 1000)",
    125: "from sklearn.datasets import load_diabetes\nfrom sklearn.linear_model import LogisticRegression\n# data = load_diabetes() # Regression data\n# Use Breast Cancer for classification usually",
    126: "from sklearn.metrics import accuracy_score\ny_pred = [0, 2, 1, 3]\ny_true = [0, 1, 2, 3]\nprint(accuracy_score(y_true, y_pred))",
    127: "from sklearn.metrics import confusion_matrix\ny_true = [1, 0, 1, 1]\ny_pred = [1, 1, 1, 0]\nprint(confusion_matrix(y_true, y_pred))",
    128: "from sklearn.naive_bayes import GaussianNB\nclf = GaussianNB()\nclf.fit([[1, 2], [2, 3]], [0, 1])",
    129: "# Visualization logic\n# plt.scatter(X_train, y_train)",
    130: "from sklearn import svm\nX = [[0, 0], [1, 1]]\ny = [0, 1]\nclf = svm.SVC()\nclf.fit(X, y)",
    131: "from sklearn.cluster import KMeans\nX = [[1, 2], [1, 4], [1, 0]]\nkmeans = KMeans(n_clusters=2).fit(X)\nprint(kmeans.labels_)",
    132: "# Elbow method loop\n# for i in range(1, 11): kmeans(i).fit(X).inertia_",
    133: "# plt.scatter(kmeans.cluster_centers_[:,0], ...)",
    134: "from sklearn.cluster import MeanShift\nms = MeanShift()\nms.fit([[1, 1], [5, 5]])",
    135: "from sklearn.cluster import estimate_bandwidth\n# bw = estimate_bandwidth(X)",
    136: "from sklearn.cluster import AgglomerativeClustering\nac = AgglomerativeClustering().fit([[1, 2], [1, 4], [1, 0]])",
    137: "from scipy.cluster.hierarchy import dendrogram, linkage\n# Z = linkage(X, 'ward')",
    138: "import nltk\n# nltk.download('punkt')",
    139: "from nltk.stem import PorterStemmer\nps = PorterStemmer()\nprint(ps.stem('running'))",
    140: "from nltk.stem import WordNetLemmatizer\n# wnl = WordNetLemmatizer()\n# wnl.lemmatize('dogs')",
    141: "import nltk\n# grammar = 'NP: {<DT>?<JJ>*<NN>}'\n# cp = nltk.RegexpParser(grammar)",
    142: "# Sentence structure logic",
    143: "# Grammar evaluation logic",
    144: "# Tree generation logic",
    145: "import cv2\nprint(cv2.__version__)",
    146: "import cv2\nimport numpy as np\n# Libraries for CV",
    147: "import cv2\n# img = cv2.imread('messi5.jpg')\n# cv2.imshow('image', img)\n# cv2.imwrite('messi5.png', img)",
    148: "# Face detection\n# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')",
    149: "# Eye detection logic",
    150: "# VideoCapture(0) frame loop",
    151: "# Live streaming detection"
}

# Values for placeholders or simple scripts are kept brief to ensure execution or simple demonstration.

json_path = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/programs.json'

try:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Update logic
    count = 0
    for item in data:
        pid = item.get('id')
        if pid in solutions:
            item['code'] = solutions[pid]
            count += 1
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Successfully updated {count} programs with solution code.")

except Exception as e:
    print(f"Error updating programs.json: {e}")
