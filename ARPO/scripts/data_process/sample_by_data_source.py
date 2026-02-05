#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从数据文件中按 data_source 类别抽取样本
"""

import pandas as pd
import argparse
import os


def sample_by_data_source(input_file, output_file, n_samples=5, random_state=42):
    """
    从输入文件中按 data_source 类别抽取指定数量的样本
    
    Args:
        input_file: 输入 parquet 文件路径
        output_file: 输出 parquet 文件路径
        n_samples: 每个类别抽取的样本数量
        random_state: 随机种子
    """
    print(f"读取文件: {input_file}")
    df = pd.read_parquet(input_file)
    
    print(f"总数据量: {len(df)}")
    print(f"\n数据源分布:")
    print(df['data_source'].value_counts())
    
    # 按 data_source 分组并抽取样本
    sampled_dfs = []
    for data_source in df['data_source'].unique():
        source_df = df[df['data_source'] == data_source].copy()
        source_count = len(source_df)
        
        # 如果该类别的样本数少于 n_samples，则全部抽取
        n_to_sample = min(n_samples, source_count)
        
        sampled = source_df.sample(n=n_to_sample, random_state=random_state)
        sampled_dfs.append(sampled)
        
        print(f"\n{data_source}: 总数={source_count}, 抽取={n_to_sample}")
    
    # 合并所有样本
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\n最终抽取的样本数: {len(result_df)}")
    print(f"\n最终数据源分布:")
    print(result_df['data_source'].value_counts())
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n创建输出目录: {output_dir}")
    
    # 保存到新文件
    print(f"\n保存到: {output_file}")
    result_df.to_parquet(output_file, index=False)
    print("完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按 data_source 类别抽取样本")
    parser.add_argument("--input", type=str, required=True, help="输入 parquet 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 parquet 文件路径")
    parser.add_argument("--n_samples", type=int, default=5, help="每个类别抽取的样本数量 (默认: 5)")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子 (默认: 42)")
    
    args = parser.parse_args()
    
    sample_by_data_source(
        input_file=args.input,
        output_file=args.output,
        n_samples=args.n_samples,
        random_state=args.random_state
    )

