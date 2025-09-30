from src.CodeParser import CodeParser
cp = CodeParser(file_extensions=["py", "js", "ts", "php", "rb", "go", "css"])
tree = cp.parse_code("def x():\n    return 1\n", "py")
print(tree.type if tree else "parse failed")
