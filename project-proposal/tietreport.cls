\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tietreport}[2026/09/11 TIET Report Class v1.1]

% =========================================================
% BASE REPORT CLASS
% =========================================================

\DeclareOption*{%
    \PassOptionsToClass{\CurrentOption}{report}
}

\ProcessOptions\relax

% openany prevents unnecessary blank pages before chapters
\LoadClass[twoside,12pt,openany]{report}

% =========================================================
% PAGE GEOMETRY
% =========================================================

\RequirePackage[
    inner=1.35in,
    outer=0.95in,
    top=1.35in,
    bottom=0.85in
]{geometry}

% =========================================================
% HEADER / FOOTER
% =========================================================

\RequirePackage{fancyhdr}

\pagestyle{fancy}

\fancyhf{}

\fancyhead[LE]{%
    \thepage\quad\@tpHeader
}

\fancyhead[RO]{%
    \@tpTitle\quad\thepage
}

\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% Chapter-opening pages
\assignpagestyle{\chapter}{empty}

% =========================================================
% CHAPTER FORMATTING
% =========================================================

\RequirePackage{titlesec}

\titleformat
    {\chapter}
    [display]
    {\raggedleft\bfseries}
    {\large Chapter \thechapter}
    {0.4em}
    {\huge}
    [\vspace{0.15em}\hrule]

% Reduced spacing to prevent excessive empty areas
\titlespacing*{\chapter}
    {0pt}
    {-25pt}
    {25pt}

% =========================================================
% REQUIRED PACKAGES
% =========================================================

\RequirePackage{hyperref}
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{amssymb}

% =========================================================
% INTERNAL TITLE-PAGE VARIABLES
% =========================================================

\newcommand{\@tpHeader}{%
    UCS 503 Software Engineering Project Report
}

\newcommand{\@tpTitle}{%
    Proposal to Final
}

\newcommand{\@tpSubTitle}{%
    How to Write a Project Report
}

\newcommand{\@tpAuthors}{%
    \texttt{102XXXXXXX} Name\\
    \texttt{102XXXXXXX} Name
}

\newcommand{\@tpSubText}{%
    BE Third Year
}

\newcommand{\@tpAdvisor}{%
    Dr. NAME OF THE ADVISOR
}

\newcommand{\@tpFooter}{%
    \large
    Computer Science and Engineering Department\\
    Thapar Institute of Engineering and Technology, Patiala
}

% =========================================================
% PUBLIC CONFIGURATION COMMANDS
% =========================================================

\newcommand{\TitlePageHeader}[1]{%
    \renewcommand{\@tpHeader}{#1}
}

\newcommand{\ProjectTitle}[1]{%
    \renewcommand{\@tpTitle}{#1}
}

\newcommand{\ProjectSubTitle}[1]{%
    \renewcommand{\@tpSubTitle}{#1}
}

\renewcommand{\author}[1]{%
    \renewcommand{\@tpAuthors}{#1}
}

\newcommand{\TitlePageSubText}[1]{%
    \renewcommand{\@tpSubText}{#1}
}

\newcommand{\AdvisorName}[1]{%
    \renewcommand{\@tpAdvisor}{#1}
}

\newcommand{\TitlePageFooterText}[1]{%
    \renewcommand{\@tpFooter}{#1}
}

% =========================================================
% CUSTOM TITLE PAGE
% =========================================================

\renewcommand{\maketitle}{%

    \begin{titlepage}

        \thispagestyle{empty}

        \centering

        \vspace*{0.4cm}

        {\large\textbf{\@tpHeader}\par}

        \vspace{1.8cm}

        {\LARGE\textbf{\@tpTitle}\par}

        \vspace{0.35cm}

        {\large\@tpSubTitle\par}

        \vspace{2.8cm}

        {\large\textbf{Submitted by:}\par}

        \vspace{0.25cm}

        {\large\@tpAuthors\par}

        \vspace{0.45cm}

        {\normalsize\@tpSubText\par}

        \vfill

        {\large\textbf{Submitted to:}\par}

        \vspace{0.25cm}

        {\normalsize\@tpAdvisor\par}

        \vspace{1.5cm}

        {\normalsize\@tpFooter\par}

    \end{titlepage}
}

% =========================================================
% END CLASS
% =========================================================

\endinput