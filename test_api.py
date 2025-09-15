
#测试用


import os
from neo4j import GraphDatabase
from raggraph import DrugGraph


def diagnose_graph(url: str, username: str, password: str) -> None:
    try:
        driver = GraphDatabase.driver(url, auth=(username, password))
        with driver.session() as session:
            # 1) 节点/关系计数
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single().get("c", 0)
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single().get("c", 0)
            print(f"[DIAG] 图规模: nodes={node_count}, rels={rel_count}")

            # 2) Top 标签分布（不依赖 APOC）
            label_stats = session.run(
                """
                MATCH (n)
                UNWIND labels(n) AS l
                RETURN l AS label, count(*) AS c
                ORDER BY c DESC
                LIMIT 10
                """
            ).data()
            if label_stats:
                print("[DIAG] Top 标签:")
                for row in label_stats:
                    print(f"  - {row['label']}: {row['c']}")
            else:
                print("[DIAG] 未检索到任何节点标签")

            # 3) 抽样三元组
            samples = session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN labels(a) AS a_labels, type(r) AS rel, labels(b) AS b_labels,
                       coalesce(a.name, a.id, a.title, a.label) AS a_name,
                       coalesce(b.name, b.id, b.title, b.label) AS b_name
                LIMIT 10
                """
            ).data()
            if samples:
                print("[DIAG] 样例三元组 (最多10条):")
                for s in samples:
                    print(f"  - ({s['a_labels']}, {s['a_name']}) -[:{s['rel']}]-> ({s['b_labels']}, {s['b_name']})")
            else:
                print("[DIAG] 未抽样到任何关系三元组")
    except Exception as e:
        print(f"[DIAG] 图诊断失败: {e}")


def main():
    # 自拟病历文本
    medical_text = ("低增生性急性白血病")
    print(medical_text)

    url = "bolt://localhost:7687"
    username = "neo4j"
    password = "12345678"

    # 图诊断
    diagnose_graph(url, username, password)

    # 调用图检索
    try:
        drug_graph = DrugGraph(
            url=url,
            username=username,
            password=password,
        )
    except Exception as e:
        print(f"[TEST] 创建 DrugGraph 实例失败: {e}")
        return

    try:
        print("[TEST] 开始基于病历文本检索图谱信息 ...")
        result = drug_graph.retrieve_medical_info(medical_text)
        print("[TEST] 检索结果:\n---\n" + str(result) + "\n---")
    except Exception as e:
        print(f"[TEST] 检索失败: {e}")


if __name__ == "__main__":
    main()
