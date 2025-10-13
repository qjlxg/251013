import pandas as pd
import json
import os

def analyze_and_recommend_funds():
    """
    该脚本读取本地文件，整合数据并筛选出符合条件的基金。
    """
    print("--- 正在加载并分析文件，为您筛选最终结果 ---")

    try:
        # 1. 读取所有文件
        selected_indices_df = pd.read_csv('selected_low_valuation_indices.csv')
        recommended_funds_df = pd.read_csv('recommended_cn_funds.csv')
        with open('comprehensive_fund_analysis.json', 'r', encoding='utf-8') as f:
            comprehensive_analysis_json = json.load(f)

        # 2. 从低估指数列表中提取行业关键词
        low_valuation_indices = selected_indices_df['指数名称'].tolist()
        keywords = []
        for index_name in low_valuation_indices:
            if '白酒' in index_name:
                keywords.append('白酒')
            if '食品' in index_name:
                keywords.append('食品')
            if '消费' in index_name:
                keywords.append('消费')
        keywords = list(set(keywords))

        if not keywords:
            print("未在低估指数文件中找到相关行业关键词（白酒、食品、消费）。")
            return

        print(f"✅ 已识别出低估行业关键词：{', '.join(keywords)}")

        # 3. 筛选出同时符合低估行业和四四三三法则的基金
        matching_funds = []
        for index, row in recommended_funds_df.iterrows():
            fund_name = row['名称']
            fund_code = row['代码']
            # 将基金代码格式化为6位数
            fund_code_str = str(fund_code).zfill(6)
            if any(k in fund_name for k in keywords):
                matching_funds.append({
                    '代码': fund_code_str,
                    '名称': fund_name,
                    '排名信息': row.to_dict()
                })

        if not matching_funds:
            print("❌ 未在推荐基金列表中找到任何与低估行业匹配的基金。")
            return

        print(f"✅ 找到 {len(matching_funds)} 只同时符合双重标准的基金。")

        # 4. 从综合分析文件中提取详细信息并整合
        analysis_data_dict = {item['fund_code']: item for item in comprehensive_analysis_json}

        final_recommendations = []
        for fund in matching_funds:
            fund_code = fund['代码']
            if fund_code in analysis_data_dict:
                details = analysis_data_dict[fund_code]

                # 提取前五大持仓
                top_holdings = [
                    {'股票名称': h.get('stock_name'), '持仓占比': h.get('proportion')}
                    for h in details.get('fund_holdings', [])[:5]
                ]

                # 提取风险指标
                risk_metrics = details.get('risk_metrics', {})

                final_recommendations.append({
                    '基金代码': fund_code,
                    '基金名称': fund['名称'],
                    '匹配行业': ', '.join([k for k in keywords if k in fund['名称']]),
                    '前五大持仓': top_holdings,
                    '年化夏普比率': f"{risk_metrics.get('sharpe_ratio', 'N/A'):.2f}" if isinstance(risk_metrics.get('sharpe_ratio'), (int, float)) else 'N/A',
                    '最大回撤': f"{risk_metrics.get('max_drawdown', 'N/A') * 100:.2f}%" if isinstance(risk_metrics.get('max_drawdown'), (int, float)) else 'N/A',
                    '基金经理': details.get('fund_managers', [])[0]['fund_managers'] if details.get('fund_managers') and details.get('fund_managers')[0] and 'fund_managers' in details.get('fund_managers')[0] else 'N/A'
                })

        # 5. 打印最终结果
        print("\n--- 最终筛选结果：可供参考和购买的基金 ---")
        if not final_recommendations:
            print("没有找到符合所有条件的基金。")
        else:
            for item in final_recommendations:
                print("\n" + "="*50)
                print(f"基金名称：{item['基金名称']} ({item['基金代码']})")
                print(f"匹配行业：{item['匹配行业']}")
                print(f"基金经理：{item['基金经理']}")
                print("核心风险指标：")
                print(f"  - 年化夏普比率：{item['年化夏普比率']}")
                print(f"  - 最大回撤：{item['最大回撤']}")
                print("前五大持仓：")
                if item['前五大持仓']:
                    for holding in item['前五大持仓']:
                        print(f"  - {holding.get('股票名称')} (占比: {holding.get('持仓占比')}%)")
                else:
                    print("  - 暂无持仓数据")

    except FileNotFoundError as e:
        print(f"错误：缺少必要的数据文件。请确保以下文件存在于同一目录下：{e.filename}")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    analyze_and_recommend_funds()
