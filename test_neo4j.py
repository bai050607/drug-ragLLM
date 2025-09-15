
#测试用


from py2neo import Graph
import matplotlib.pyplot as plt
import networkx as nx
from py2neo import Node, Relationship
#本文件用于测试连接
graph = Graph("bolt://127.0.0.1:7687", auth=("neo4j", "12345678"))

try:
    result = graph.run("MATCH (n) RETURN count(n) as node_count")
    print(f"数据库中共有 {result.data()[0]['node_count']} 个节点")
except Exception as e:
    print(f"连接失败: {e}")

