

[TOC]

# Vim常用操作

## 概念

**Vim**作为**Linux**系统自带的文本编辑器，掌握如何使用是很有必要的。



## Vim模式

* **Normal** 模式： 进入Vim后的一般模式
* **Insert** 模式： 按下i后可以进入插入模式，像普通编辑器一样可以写入文件
* **Visual** 模式： 按下v后进入选择模式，可以选择文件内容



## Vim打开和切换文件

1. 终端`vim file1 file2 ...`可以打开多个文件。
2. `:ls`显示打开的文件，可以使用`:bn`在文件间切换( n也可以换成`:ls`里给出的文件序号 )。
3. 打开vim后，可以用`:e fileName`来打开文件或者新建文件。
4. 在终端`vim -o file1 file2 ...`可以打开多个文件(横向分隔屏幕)。
5. 终端`vim -O file1 file2 ...`可以打开多个文件(纵向分隔屏幕)。
6. `Ctrl`+`w`+`w`在窗口间切换光标，第二个`w`也可以用`h、j、k、l`来光标代表移动方向。



## Vim退出

1. `:q`：退出。
2. `:q!`：强制退出，放弃所有修改。
3. `:wq`：保存修改并退出。



## 常用快捷键

1. 方向键←↓↑→也可以用`h、j、k、l`。
2. `0`到行首，`$`到行尾。
3. `gg`到文档首行，`G`到文档结尾。
4. `Ctrl`+`f`下一页，`Ctrl`+`b`上一页。
5. `Ctrl`+`u`往上半页，`Ctrl`+`d`往下半页。
6. `w`或`e`光标往后跳一个单词，`b`光标往前跳一个单词。
7. `:98`跳转到第98行。
8. `q:`显示**命令行历史记录**窗口。
9. `!bash_Command`不退出vim暂时返回终端界面执行该命令。
10. `H`将光标移动到屏幕首行，`M`将光标移动到屏幕中间行，`L`将光标移动到屏幕最后一行。



## Vim分屏

1. `:sp`或者`:split`横向分隔屏幕，后面可以加文件名。
2. `:vs`或者`:vsplit`纵向分隔屏幕，后面可以加文件名。
3. `:only`只保留光标所在分屏，关闭其他分屏。
4. 在**nerdtree**插件中，选中文件后按`s`纵向分隔屏幕，按`i`水平分隔屏幕。



## 在窗口间游走

在gvim或vim中，在窗口中移动其实非常简单，因为gvim已默认支持鼠标点击来换编辑窗口，而vim中，则可以打开mouse选项，
`:set mouse = a`  为命令、输入、导航都激活鼠标的使用

我们知道vim的特色就是可以脱离鼠标而工作，所以可以使用vim提供的全套导航命令，在会话中快速而准确的移动编辑窗口。

按住`Ctrl + W`，然后再加上**h, j, k, l**，分别表示向左、下、上、右移动窗口
`Ctrl + w + h：`向左移动窗口
`Ctrl + w + j：` 向下移动窗口
`Ctrl + w + j：` 向上移动窗口
`Ctrl + w + l：` 向右移动窗口

`Ctrl + w + w：`这个命令会在所有窗口中循环移动
`Ctrl + w + t：`移动到最左上角的窗口
`Ctrl + w + b：`移动到最右下角的窗口
`Ctrl + w + p：`移动到前一个访问的窗口



## 复制粘贴

- 在**Visual**模式下选择文档内容后按`y`键，复制被选择内容。
- 按`p`键粘贴，注意粘贴从**紧跟光标后的那个字符**之后才开始。

> abc | d **"COPIED_TEXT"** efghk... ( | 是光标)

- 选择内容后按`d`删除或者剪贴。
- `yy`复制当前行，`dd`删除(剪贴)当前行。



## 查找和替换

### 查找

1. 在**Normal**模式下，按`/`进入查找模式，输入`/word`后回车，高亮显示所有文档`word`，按`n`跳到下一个`word`,按`N`跳到上一个。
2. 若输入`/word\c`代表大小写不敏感查找，`\C`代表大小写敏感。
3. 输入`:noh`取消高亮。
4. 按下`*`高亮查找光标位置处的单词，但若查找`word`,`helloword`中的`word`不会被高亮。
5. 按下`g*`高亮查找光标位置处的单词，若查找`word`,`helloword`中的`word`也会被高亮。
6. 在**Normal**模式下按`q`+`/`显示**查找历史记录**窗口。

### 替换

1. `:s/word/excel`：替换当前行所有`word`为`excel`。
2. `:s/word/excel/g`：替换当前行第一个`word`为`excel`,`/g`代表只替换每行第一个。
3. `:%s/word/excel`：替换全文所有`word`为`excel`。
4. `:%s/word/excel/gc`：其中`/c`代表需要确认，并提示：`replace with excel (y/n/a/q/l/^E/^Y)?`，其中`a`表示替换所有，`q`表示退出查找模式， `l`表示替换当前位置并退出，`^E`与`^Y`是光标移动快捷键。
5. `:2,11s/word/excel`：替换第2到11行的`word`为`excel`。
6. `:.,$s/word/excel`：替换当前行到最后一行的`word`为`excel`，`.`代表当前行，`$`代表最后一行。
7. `:.,+2s/word/excel`：替换当前行与接下来2行的`word`为`excel`。
8. 在**Visual**模式下选择后按`:`, Vim自动补全为`:'<,'>`,然后手动补全`:'<,'>s/word/excel`，将选择区域的`word`替换为`excel`。
9. `:s/word/excel/i`：`/i`代表大小不敏感查找，等同于`:s/word\c/excel`，而`/I`代表大小写敏感查找。



## 代码折叠

代码折叠是**Vim**的高级功能，即便没掌握也不影响**Vim**的正常使用。

- `set foldenable`: 打开代码折叠功能，可以写在根目录下的**.vimrc**文件里。
- `set foldmethod=syntax`: 设置折叠方式(`foldmethod`可缩写为`fdm`)，常用的折叠方式有:

> syntax: 按 C \ C++ 语法折叠 {};
> indent: 按缩进折叠，适用于 Python 。

- `set foldlevelstart=99`: 打开文件时不自动折叠代码。
- `zc` & `zo`: 在**Normal**模式下将光标移动到代码的可折叠位置，按`zc`折叠代码(close)，`zo`打开折叠的代码(open)。

***小技巧*** : 将`nnoremap <space> @=((foldclosed(line('.')) < 0) ? 'zc' : 'zo')<CR>`写入.**vimrc**文件，可将`zc`和`zo`映射为空格键。



## 拼写检查

对于英文单词的拼写，**Vim** 可以自动检查拼写 (**Spell Checking**)。

1. `set spell`: 打开拼写检查，可以写在根目录下的 **.vimrc** 文件里，不过更推荐在需要使用时在 **Vim** 中手动打开 `:set spell`。
2. 在 **.vimrc** 文件里写入 `inoremap <C-l> <c-g>u<Esc>[s1z=`]a<c-g>u` 来将 `Ctrl+l` 映射为快捷键，其中 `[s` 代表光标跳到上一个拼写错误处， `1z=` 选择第一个推荐的正确拼写， ``]a` 光标跳回原来位置。设置完就可以通过 `Ctrl+l` 快速改正单词。



## 其他代码

* **Normal** 模式
  1.  `u`表示 撤回  

  2.    `y`是复制

  3. ` yy`是复制一行

  4.  `p`是粘贴

  5. ` d`是删除

  6.  全局替换 `% s/java/python/g`   全局替换java

* **Visual** 模式
  1. `V`选择行
  
  2. `ctrl+v`长方形块状选择



## 快速纠错

* 插入模式

  1. `ctrl+H` 删除上一个字符

  2. `ctrl+W` 删除上一个单词

  3. `ctrl+U`删除上一行



* 切换模式
  1. 可以使用`ctrl+[`代替`esc`切换模式
  2. 或者在配置文件中配置`jj`代替

* **Normal**

  1. `gi`可以快速进入上一次编辑的位置

  2. `w`移动下一个单词的开头

  3. `e`移动到下一个单词的结尾

  4. `f{char} `行间搜索词 分号`;`继续往下搜索  逗号，往前查找

* **Command**命令行
1. `:set nu` 设置行号
  
2. `:syntax on `语法高亮



## 行内移动

1. `0`移动到行首第一个字符

2. `$`移动到行尾

3. `gg`移动文件开头

4. `G`移动到文件结尾

5. `ctrl+o`快速返回

6. `H/M/L` 移动到屏幕开头 中间 结尾

7. `ctrl+u/ctrl+f `上下页翻页  zz把屏幕置为中间



## 快速增删改查

* **Normal**模式下 

  1. `x`删除一个字符
  2. `nx`删除n个字符
  3. `dw`删除一个单词
  4. `dd`删除一行
  5. `ndd `删除n行
  6. `d$`删除到行尾 
  6. `q:`查看历史命令
  
  

## 快速修改

1. `r`可以替换一个字符 
2. `R`直接进入替换模式           
3. `s`删除当前字符并且进入插入模式
4. `c`配合文本对象，快速进行修改
5. `ct”`  直接删除到冒号并且变成插入模式
6. `/`进行查找
7. `n/N `继续上一次/下一次的查找



## 多文件操作

* buffer的切换

  1. `:ls `列举当前的缓冲区 
  2. `:b n`跳转到第n个缓冲区
  3. `:e a.v`打开文件
  4. `:tabn` 切换到下一个tab 也可以在后面加入数字
  5. `:tabnew a.v` 使用tab的形式打开一个文件
  6. `:tabc `关闭当前标签页

* 次数+示例+文本对象

  1. `iw `表示inner word    viw选中一个单词
  2. `aw`表示around word vaw选中单词还有周围的空格
  3. `d`表示删除操作
  4. 前面可以加入数字表示几个单词
  5. `vi”` 表示快速选中双引号的内容
  6. `ci”`删除双引号里面的内容并进行插入模式
  7. `ci(`
  8. `ci{`
  9. `:set paste `粘贴不会出错

  

## 补全

1. `ctrl+n ctrl+p`补全单词

2. `ctrl+x ctrl+f` 补全文件名

3. `ctrl+x `补全代码 需要开启文件类型检查，安装插件



## 配色

1. `:colorscheme +ctrl+d`

2. `:colorscheme+主题`




## 插件

> 查找插件

[Vim Awesome](https://vimawesome.com/)

> 修改启动界面：vim-startify

[mhinz/vim-startify: The fancy start screen for Vim. (github.com)](https://github.com/mhinz/vim-startify)

> 状态栏美化：vim-airline

[vim-airline/vim-airline: lean & mean status/tabline for vim that's light as air (github.com)](https://github.com/vim-airline/vim-airline)

> 增加代码缩进线条：indentline

[Yggdroot/indentLine: A vim plugin to display the indention levels with thin vertical lines (github.com)](https://github.com/Yggdroot/indentLine)



### 配色方案

> vim-hybrid

[w0ng/vim-hybrid: A dark color scheme for Vim (github.com)](https://github.com/w0ng/vim-hybrid)

> vim-colors-solarized

[altercation/vim-colors-solarized: precision colorscheme for the vim text editor (github.com)](https://github.com/altercation/vim-colors-solarized)

> gruvbox

[morhetz/gruvbox: Retro groove color scheme for Vim (github.com)](https://github.com/morhetz/gruvbox)’



### 文件管理器

> nerdtree
>

[preservim/nerdtree: A tree explorer plugin for vim. (github.com)](https://github.com/preservim/nerdtree)

1. `ctrl+w+p`  从目录跳转到文件（上一个窗口）

2. `<leader>+f `从文件跳转回去目录 （映射）



### ctrlp插件模糊搜索文件

> ctrlp.vim

[kien/ctrlp.vim: Fuzzy file, buffer, mru, tag, etc finder. (github.com)](https://github.com/kien/ctrlp.vim)

* ctrl P模糊查找

[(1 封私信 / 80 条消息) 如何将 vi 中的内容复制到 Windows，以及 Windows 中的内容复制到 vi？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/22504275#:~:text=官方提供的 vim.exe 能够通过 `%2B` 跟 `*` 寄存器访问 windows,Shift-Insert 直接在插入模式下将剪贴板内容贴到 vim 中，会受到 indent 等设置的影响，set paste 可以关闭不需要的缩进。)



### 文件快速定位插件

>  vim-easymotion

* 映射之后双击ss

[easymotion/vim-easymotion: Vim motions on speed! (github.com)](https://github.com/easymotion/vim-easymotion)



### 成对编辑双引号

> vim-surround

[tpope/vim-surround: surround.vim: Delete/change/add parentheses/quotes/XML-tags/much more with ease (github.com)](https://github.com/tpope/vim-surround)

* normal模式下增加，删除，修改成对内容

  1.  `ds " `(ds delete )
  2.  `cs " '` (cs change )
  3.  `ys iw "` (ys add)

  

### 模糊搜索文件内容

> fzf命令行搜索工具
>

[junegunn/fzf.vim: fzf vim (github.com)](https://github.com/junegunn/fzf.vim)

1. Ag [PATTERN] 模糊搜索字符串

2. Files [PATH] 模糊搜索目录

```sh
Plug 'junegunn/fzf.vim'
Plug 'junegunn/fzf', { 'dir':'~/.fzf','do': './install --all' }
```

[终端命令工具收集fd fzf - hongdada - 博客园 (cnblogs.com)](https://www.cnblogs.com/hongdada/p/14071281.html)



### 搜索替换插件

> far.vim

[brooth/far.vim: Find And Replace Vim plugin (github.com)](https://github.com/brooth/far.vim)

```
:Far foo bar **/*.py
:Fardo
```

### python-mode

> python-mode

[python-mode/python-mode: Vim python-mode. PyLint, Rope, Pydoc, breakpoints from box. (github.com)](https://github.com/python-mode/python-mode)



### 高亮

> vim-interestingwords

[lfv89/vim-interestingwords: ☀️ A vim plugin for highlighting and navigating through different words in a buffer. (github.com)](https://github.com/lfv89/vim-interestingwords)

```sh
Plug 'lfv89/vim-interestingwords'
```



### 代码补全

> deoplete.nvim

[Shougo/deoplete.nvim: Dark powered asynchronous completion framework for neovim/Vim8 (github.com)](https://github.com/Shougo/deoplete.nvim)

[deoplete-plugins/deoplete-jedi: deoplete.nvim source for Python (github.com)](https://github.com/deoplete-plugins/deoplete-jedi)

> coc.nvim

[neoclide/coc.nvim: Nodejs extension host for vim & neovim, load extensions like VSCode and host language servers. (github.com)](https://github.com/neoclide/coc.nvim)



### 格式化和静态检查

> vim-autoformat

[sbdchd/neoformat: A (Neo)vim plugin for formatting code. (github.com)](https://github.com/sbdchd/neoformat)

```python
pip install autopp8
```

> Neoformat

[sbdchd/neoformat: A (Neo)vim plugin for formatting code. (github.com)](https://github.com/sbdchd/neoformat)

静态检查Lint

> ale

```python
pip install pylint
```

[dense-analysis/ale: Check syntax in Vim asynchronously and fix files, with Language Server Protocol (LSP) support (github.com)](https://github.com/dense-analysis/ale#usage-linting)

> neomake

 [neomake/neomake: Asynchronous linting and make framework for Neovim/Vim (github.com)](https://github.com/neomake/neomake)



### 快速注释代码

> vim-commentary

[tpope/vim-commentary: commentary.vim: comment stuff out (github.com)](https://github.com/tpope/vim-commentary)

1. `gcc`注释本行

2. `gcgc`取消本行注释

3. `v`选中`gc`注释和取消注释



## Tmux

ctrl+B+%分屏



## 开源的配置

> SpaceVim

[SpaceVim/SpaceVim: A community-driven modular vim/neovim distribution - The ultimate vimrc (github.com)](https://github.com/SpaceVim/SpaceVim)

> vim-config

[rafi/vim-config: Lean mean Neovim machine, carefully crafted with Use with latest Neovim. (github.com)](https://github.com/rafi/vim-config)



## 个人配置

```sh
set number
syntax on
" hybrid theme setings
set background=dark
colorscheme hybrid

let mapleader=','
inoremap <leader>w <Esc>:w<cr>
inoremap jj <Esc>
noremap <C-h> <C-w>h
noremap <C-j> <C-w>j
noremap <C-k> <C-w>k
noremap <C-l> <C-w>l
com!FormatJSON %!python3 -m json.tool

call plug#begin()
Plug 'tpope/vim-sensible'
Plug 'mhinz/vim-startify'
Plug 'preservim/nerdtree'
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
Plug 'yggdroot/indentline'
Plug 'w0ng/vim-hybrid'
Plug 'ctrlpvim/ctrlp.vim'
Plug 'easymotion/vim-easymotion'
Plug 'tpope/vim-surround'
Plug 'junegunn/fzf', { 'dir':'~/.fzf','do': './install --all' }
Plug 'junegunn/fzf.vim'
Plug 'brooth/far.vim'
Plug 'python-mode/python-mode', { 'for': 'python', 'branch': 'develop' }
Plug 'lfv89/vim-interestingwords'
" format
Plug 'sbdchd/neoformat'
Plug 'tpope/vim-commentary'
let g:deoplete#enable_at_startup = 1
call plug#end()

" nerdtree settings
nnoremap <leader>f :NERDTreeFind<cr>
nnoremap <leader>t :NERDTreeToggle<cr>
" Start NERDTree and leave the cursor in it.
" autocmd VimEnter * NERDTree
" set width
let NERDTreeWinSize=31
" show hidden files
let NERDTreeShowHidden=1
" not show files listed
let NERDTreeIgnore= ['\.git$', '\.hg$', '\.svn$', '\.stversions$', '\.pyc$', '\.pyo$', '\.swp$','\.DS_Store$', '\.sass-cache$', '__pycache__$', '\.egg-info$', '\.ropeproject$',]
" map the ctrlp
let g:ctrlp_map = '<c-p>'
nmap ss <Plug>(easymotion-s2)
" python mode
let g:pymode_python= 'python3'
let g:pymode_trim_whitespaces=1
let g:pymode_doc_bind='K'
let g:pymode_rope_goto_definition_bind="<C-]>"
let g:pymode_lint=1
let g:pymode_lint_checkers=[ 'pyflakes', 'pep8', 'mccabe', 'pylint']
let g:pymode_options_max_line_length=120
```

## IdeaVim

### 插件AceJump

**[AceJump](https://github.com/acejump/AceJump)**

```sh
" Press `f` to activate AceJump
map f <Action>(AceAction)
" Press `F` to activate Target Mode
map F <Action>(AceTargetAction)
" Press `ff` to activate Line Mode          #跳转到开头和结尾
map ff <Action>(AceLineAction)
```

## **11 代码折叠**

```text
zo - 打开折叠
zc - 关闭折叠
```

比较详细的ideavim配置：[(214条消息) IdeaVim 史诗级分享_其樂无穷的博客-CSDN博客](https://blog.csdn.net/qq_25955145/article/details/109914087)

### 文件树

[NERDTree support · JetBrains/ideavim Wiki (github.com)](https://github.com/JetBrains/ideavim/wiki/NERDTree-support)

[(214条消息) VIM插件：目录导航与操作插件NERDTree的使用方法_vim nerdtree_嵌入式技术的博客-CSDN博客](https://blog.csdn.net/weixin_37926734/article/details/124919260)

```
:NERDTree
o open/close文件
在目录中 q可以直接关闭目录
p在目录中是走到父目录中去
s分屏打开文件
```

保持光标位置不变（不变是相对的，当光标所在行超出光标可活动行范围时，光标保持在最上/最下可活动行）移动屏幕：向上翻页ctrl + y，向下翻页ctrl + e。
滚动半屏：向上滚动半屏ctrl + u，向下滚动半屏ctrl + d。
滚动一屏：向上滚动一屏ctrl + b，向下滚动一屏ctrl + f。
