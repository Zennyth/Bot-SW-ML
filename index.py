# coding: utf-8

import cgi 

form = cgi.FieldStorage()
print("Content-type: text/html; charset=utf-8\n")

if form.getvalue("start"):
	print("Start")



html = """<!DOCTYPE html>
<head>
    <title>Mon programme</title>
</head>
<body>
    <form action="/index.py" method="post">
        <input type="submit" name="start" value="Start">
    </form> 
</body>
</html>
"""

print(html)