# 定义你想测试的 beta 值列表
beta_values=(1.351)

# 循环遍历列表中的每个 beta 值
for beta_val in "${beta_values[@]}"
do
  echo "------ Running with beta = ${beta_val} ------"
  # 调用你的脚本，并将当前的 beta_val 作为第 7 个参数传递
  # 注意：确保其他参数 $1 到 $5 和 $8 (temp=1) 是正确的
  ./InfKV_head_base.sh 0 InfKV 128 flash_attention_2 meta-llama/Meta-Llama-3-8B-Instruct ${beta_val} 1
  echo "------ Finished beta = ${beta_val} ------"
  echo # 添加一个空行，方便区分不同运行的输出
done

echo "All beta values processed."