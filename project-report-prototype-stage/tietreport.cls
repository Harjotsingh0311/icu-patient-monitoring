%% ============================================================
%% tietreport.cls
%% TIET Software Engineering Project Report Class
%% Version 1.1 -- Blank-page corrected
%% ============================================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tietreport}[2026/09/11 TIET Project Report Class]

%% ============================================================
%% BASE CLASS
%% ============================================================

%% IMPORTANT:
%% openany prevents LaTeX from inserting blank pages before
%% chapters when using a two-sided document.

\DeclareOption*{%
    \PassOptionsToClass{\CurrentOption}{report}%
}

\ProcessOptions\relax

\LoadClass[
    12pt,
    twoside,
    openany
]{report}

%% ============================================================
%% PAGE GEOMETRY
%% ============================================================

\RequirePackage[
    inner=1.5in,
    outer=1in,
    top=1.8in,
    bottom=0.85in,
    headheight=15pt,
    headsep=0.25in
]{geometry}

%% ============================================================
%% FONT / TYPOGRAPHY
%% ============================================================

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{lmodern}
\RequirePackage{microtype}
\RequirePackage{setspace}

%% ============================================================
%% HEADER AND FOOTER
%% ============================================================

\RequirePackage{fancyhdr}

\pagestyle{fancy}

\fancyhf{}

%% Even pages
\fancyhead[LE]{%
    \thepage\quad\@tpHeader
}

%% Odd pages
\fancyhead[RO]{%
    \@tpTitle\quad\thepage
}

\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0pt}

%% Chapter-opening pages
\fancypagestyle{plain}{%
    \fancyhf{}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
}

%% ============================================================
%% CHAPTER FORMATTING
%% ============================================================

\RequirePackage{titlesec}

\titleformat
    {\chapter}
    [display]
    {\raggedleft\bfseries}
    {\large Chapter \thechapter}
    {0.3em}
    {\huge}
    [\vspace{0.1em}\hrule]

\titlespacing*{\chapter}
    {0pt}
    {-35pt}
    {30pt}

\assignpagestyle{\chapter}{plain}

%% ============================================================
%% STANDARD PACKAGES
%% ============================================================

\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{float}
\RequirePackage{caption}
\RequirePackage{enumitem}
\RequirePackage{xcolor}

%% ============================================================
%% HYPERREF
%% ============================================================

\RequirePackage[
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=black,
    bookmarks=true,
    bookmarksnumbered=true,
    pdfstartview=FitH
]{hyperref}

%% ============================================================
%% TITLE PAGE VARIABLES
%% ============================================================

\newcommand{\@tpHeader}{%
    UCS 503 -- Software Engineering Project Report%
}

\newcommand{\@tpTitle}{%
    ICU Patient Face Monitoring System%
}

\newcommand{\@tpSubTitle}{%
    A Computer-Vision-Based Non-Contact System for Continuous
    Facial and Ocular Monitoring of Intensive Care Unit Patients%
}

\newcommand{\@tpAuthors}{%
    \texttt{1024240039} Harjot Singh\\
    \texttt{1024240103} Pahul Singh%
}

\newcommand{\@tpSubText}{%
    BE Third Year, Data Science and Artificial Intelligence%
}

\newcommand{\@tpAdvisor}{%
    Ms. Anushka%
}

\newcommand{\@tpFooter}{%
    \large Project Report\\[0.2cm]
    \bfseries Computer Science and Engineering Department\\[0.15cm]
    Thapar Institute of Engineering and Technology, Patiala%
}

%% ============================================================
%% PUBLIC CONFIGURATION COMMANDS
%% ============================================================

\newcommand{\TitlePageHeader}[1]{%
    \renewcommand{\@tpHeader}{#1}%
}

\newcommand{\ProjectTitle}[1]{%
    \renewcommand{\@tpTitle}{#1}%
}

\newcommand{\ProjectSubTitle}[1]{%
    \renewcommand{\@tpSubTitle}{#1}%
}

\renewcommand{\author}[1]{%
    \renewcommand{\@tpAuthors}{#1}%
}

\newcommand{\TitlePageSubText}[1]{%
    \renewcommand{\@tpSubText}{#1}%
}

\newcommand{\AdvisorName}[1]{%
    \renewcommand{\@tpAdvisor}{#1}%
}

\newcommand{\TitlePageFooterText}[1]{%
    \renewcommand{\@tpFooter}{#1}%
}

%% ============================================================
%% TITLE PAGE
%% ============================================================

\renewcommand{\maketitle}{%

    \begin{titlepage}

        \thispagestyle{empty}

        \centering

        \vspace*{0.6cm}

        {\large\bfseries
            \@tpHeader
        }

        \vspace{2.0cm}

        {\LARGE\bfseries
            \@tpTitle
        }

        \vspace{0.4cm}

        {\large
            \@tpSubTitle
        }

        \vspace{3.0cm}

        {\large\bfseries
            Submitted by:
        }

        \vspace{0.4cm}

        {\large
            \@tpAuthors
        }

        \vspace{0.7cm}

        {\normalsize
            \@tpSubText
        }

        \vfill

        {\large\bfseries
            Submitted to:
        }

        \vspace{0.35cm}

        {\normalsize
            \@tpAdvisor
        }

        \vspace{1.8cm}

        {\normalsize
            \@tpFooter
        }

    \end{titlepage}

    %% Do NOT use \cleardoublepage here.
    %% openany already prevents artificial blank pages.

    \clearpage
}

%% ============================================================
%% TABLE OF CONTENTS
%% ============================================================

\setcounter{tocdepth}{2}

%% ============================================================
%% END
%% ============================================================

\endinput