import os
import json
import pandas as pd

def aggregate_results_from_flat_dir(results_dir="all_my_results"):
    """
    在一个扁平化的目录中查找所有summary_*.json文件，
    提取超参数和评估结果，并汇总成一个Pandas DataFrame。
    """
    all_results = []
    
    # 检查结果目录是否存在
    if not os.path.isdir(results_dir):
        print(f"错误：目录 '{results_dir}' 不存在。")
        return pd.DataFrame()

    # 1. 直接列出目录下的所有文件
    all_files = os.listdir(results_dir)
    
    # 2. 筛选出所有的summary文件
    summary_files = [f for f in all_files if f.startswith("summary_") and f.endswith(".json")]
    
    if not summary_files:
        print(f"在目录 '{results_dir}' 中未找到任何 'summary_*.json' 文件。")
        return pd.DataFrame()

    print(f"找到了 {len(summary_files)} 个summary文件，开始处理...")

    # 3. 遍历每一个summary文件
    for summary_file_name in summary_files:
        summary_file_path = os.path.join(results_dir, summary_file_name)
        
        try:
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            hyperparameters = data.get("hyperparameters", {})
            eval_results = data.get("evaluation_results", {})
            
            # 将列表类型的超参数转换为字符串，以便在CSV中显示
            for key, value in hyperparameters.items():
                if isinstance(value, list):
                    hyperparameters[key] = str(value)
            
            combined_data = {**hyperparameters, **eval_results}
            # 使用文件名中的时间戳作为唯一ID
            run_id = summary_file_name.replace("summary_", "").replace(".json", "")
            combined_data['run_id'] = run_id
            
            all_results.append(combined_data)
            
        except Exception as e:
            print(f"错误：处理文件 '{summary_file_name}' 失败 - {e}")

    if not all_results:
        print("未能成功处理任何有效的实验结果。")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_results)
    return df

if __name__ == "__main__":
    # 只需要设置这一个文件夹路径即可！
    flat_results_directory = "rag_results"
    
    results_df = aggregate_results_from_flat_dir(results_dir=flat_results_directory)
    
    if not results_df.empty:
        output_csv_path = "aggregated_rag_results_flat.csv"
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n聚合完成！所有结果已保存到: {output_csv_path}")
        print("\n聚合数据预览：")
        print(results_df.head())