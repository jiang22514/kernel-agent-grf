# 图1重绘说明与检查

本图为方法示意图，无新拟合、模拟、API请求或实验数据。只输出在本修订目录，未覆盖投稿原图，也未改TeX。

## 表达与方法边界

- 上层橙色回路：LLM提出核–均值候选及2个备用项，程序验证语法/去重，只评分前b个合法未访问候选；总访问预算仍为20。4×5和5×4表示两种既有批次安排。
- 输入目录为unranked（未按评分排名），不暗示其ID顺序被随机打乱。反馈包含AIC、相对已访问最优值的差、参数个数k、拟合状态和最佳已访问结构的邻域。
- 数值评分器拟合GP参数；本研究基准使用预拟合同一分数缓存，图中文字明确为cached scores。LLM不拟合连续参数。
- 下层保留协方差分解、坐标并集上的独立先验分量求和、联合Matheron条件化和弱区统计。
- 灰色虚线旁路为全数据固定参考结构及留整孔参数重拟合，未画成每折重跑agent选核。

## 交付

- figures/overall_framework_b7.pdf：矢量投稿图，183×145 mm。
- figures/overall_framework_b7.svg：可编辑SVG文字。
- figures/overall_framework_b7.png：600 dpi，4322×3425像素。
- figures/overall_framework_b7_pdf_preview.png：由PDF渲染的300 dpi预览。
- draw_agent_framework.py：完整Matplotlib源码。
- figure_contract.md、figure_qa.json、source_preflight.json：图形契约与检查记录。

## 检查结果

实际PDF最小字号9 pt，Arial字体嵌入、文字可选择；SVG含36个可编辑文字节点。绘图程序逐文字检查节点边界、画布裁切、文字重叠，均通过。已人工检查最终PDF预览，闭环箭头及下层旁路清楚，未发现遮挡或溢出。

科研图源检查13项PASS、0项FAIL、1项WARN。唯一WARN为未输出TIFF；本轮明确要求PDF/SVG/PNG，矢量PDF为投稿主文件，故未额外生成TIFF。

图注可进一步说明b为单批评分候选数（4或5）、k为拟合参数个数，以及全数据比较所用的具体模拟引擎。
