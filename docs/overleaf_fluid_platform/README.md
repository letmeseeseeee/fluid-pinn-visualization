# Overleaf 使用说明

这个目录已经整理成可直接上传到 Overleaf 的论文工程，包含：

- `main.tex`：主 LaTeX 文件
- `refs.bib`：参考文献 BibTeX 文件
- `oucart.cls`：中国海洋大学本科毕业论文模板类文件
- `assets/`：封面、摘要、签名页样式与校徽文件
- `figures/`：论文插图

建议在 Overleaf 中使用以下编译方式：

1. 编译器选择 `XeLaTeX`
2. 参考文献工具选择 `BibTeX`
3. 编译顺序通常为：
   - `XeLaTeX`
   - `BibTeX`
   - `XeLaTeX`
   - `XeLaTeX`

如果 Overleaf 提示参考文献样式错误，请确认：

1. 主文件是 `main.tex`
2. 编译器是 `XeLaTeX`
3. 参考文献样式已经使用 `gbt7714-numerical`

当前工程已经套用 OUC-LaTeX-bachelor 项目的本科论文模板，封面、签名页、摘要、目录、正文标题、图表编号、参考文献和致谢目录项均由模板控制。
