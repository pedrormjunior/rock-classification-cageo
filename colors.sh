# Copyright (c) 2019-2023
# Shell-ML Project
# Pedro Ribeiro Mendes Júnior <pedrormjunior@gmail.com> et al.
# Artificial Intelligence Lab. Recod.ai
# Institute of Computing (IC)
# University of Campinas (Unicamp)
# Campinas, São Paulo, Brazil
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# Based on https://stackoverflow.com/a/16844327
RCol='\e[0m'    # Text Reset

# Regular           Bold                Underline           High Intensity      BoldHigh Intens     Background          High Intensity Backgrounds
Bla='\e[0;30m';     BBla='\e[1;30m';    UBla='\e[4;30m';    IBla='\e[0;90m';    BIBla='\e[1;90m';   On_Bla='\e[40m';    On_IBla='\e[0;100m';
Red='\e[0;31m';     BRed='\e[1;31m';    URed='\e[4;31m';    IRed='\e[0;91m';    BIRed='\e[1;91m';   On_Red='\e[41m';    On_IRed='\e[0;101m';
Gre='\e[0;32m';     BGre='\e[1;32m';    UGre='\e[4;32m';    IGre='\e[0;92m';    BIGre='\e[1;92m';   On_Gre='\e[42m';    On_IGre='\e[0;102m';
Yel='\e[0;33m';     BYel='\e[1;33m';    UYel='\e[4;33m';    IYel='\e[0;93m';    BIYel='\e[1;93m';   On_Yel='\e[43m';    On_IYel='\e[0;103m';
Blu='\e[0;34m';     BBlu='\e[1;34m';    UBlu='\e[4;34m';    IBlu='\e[0;94m';    BIBlu='\e[1;94m';   On_Blu='\e[44m';    On_IBlu='\e[0;104m';
Pur='\e[0;35m';     BPur='\e[1;35m';    UPur='\e[4;35m';    IPur='\e[0;95m';    BIPur='\e[1;95m';   On_Pur='\e[45m';    On_IPur='\e[0;105m';
Cya='\e[0;36m';     BCya='\e[1;36m';    UCya='\e[4;36m';    ICya='\e[0;96m';    BICya='\e[1;96m';   On_Cya='\e[46m';    On_ICya='\e[0;106m';
Whi='\e[0;37m';     BWhi='\e[1;37m';    UWhi='\e[4;37m';    IWhi='\e[0;97m';    BIWhi='\e[1;97m';   On_Whi='\e[47m';    On_IWhi='\e[0;107m';
Reg='\e[0m';        Bold='\e[1m';       Unde='\e[4m';

# Text Reset
function echo_RCol() { echo -e "${RCol}$@${Reg}"; };
# Regular
function echo_Bla() { echo -e "${Bla}$@${Reg}"; };
function echo_Red() { echo -e "${Red}$@${Reg}"; };
function echo_Gre() { echo -e "${Gre}$@${Reg}"; };
function echo_Yel() { echo -e "${Yel}$@${Reg}"; };
function echo_Blu() { echo -e "${Blu}$@${Reg}"; };
function echo_Pur() { echo -e "${Pur}$@${Reg}"; };
function echo_Cya() { echo -e "${Cya}$@${Reg}"; };
function echo_Whi() { echo -e "${Whi}$@${Reg}"; };
function echo_Reg() { echo -e "${Reg}$@${Reg}"; };
# Bold
function echo_BBla() { echo -e "${BBla}$@${Reg}"; };
function echo_BRed() { echo -e "${BRed}$@${Reg}"; };
function echo_BGre() { echo -e "${BGre}$@${Reg}"; };
function echo_BYel() { echo -e "${BYel}$@${Reg}"; };
function echo_BBlu() { echo -e "${BBlu}$@${Reg}"; };
function echo_BPur() { echo -e "${BPur}$@${Reg}"; };
function echo_BCya() { echo -e "${BCya}$@${Reg}"; };
function echo_BWhi() { echo -e "${BWhi}$@${Reg}"; };
function echo_Bold() { echo -e "${Bold}$@${Reg}"; };
# Underline
function echo_UBla() { echo -e "${UBla}$@${Reg}"; };
function echo_URed() { echo -e "${URed}$@${Reg}"; };
function echo_UGre() { echo -e "${UGre}$@${Reg}"; };
function echo_UYel() { echo -e "${UYel}$@${Reg}"; };
function echo_UBlu() { echo -e "${UBlu}$@${Reg}"; };
function echo_UPur() { echo -e "${UPur}$@${Reg}"; };
function echo_UCya() { echo -e "${UCya}$@${Reg}"; };
function echo_UWhi() { echo -e "${UWhi}$@${Reg}"; };
function echo_Unde() { echo -e "${Unde}$@${Reg}"; };
# High Intensity
function echo_IBla() { echo -e "${IBla}$@${Reg}"; };
function echo_IRed() { echo -e "${IRed}$@${Reg}"; };
function echo_IGre() { echo -e "${IGre}$@${Reg}"; };
function echo_IYel() { echo -e "${IYel}$@${Reg}"; };
function echo_IBlu() { echo -e "${IBlu}$@${Reg}"; };
function echo_IPur() { echo -e "${IPur}$@${Reg}"; };
function echo_ICya() { echo -e "${ICya}$@${Reg}"; };
function echo_IWhi() { echo -e "${IWhi}$@${Reg}"; };
# BoldHigh Intens
function echo_BIBla() { echo -e "${BIBla}$@${Reg}"; };
function echo_BIRed() { echo -e "${BIRed}$@${Reg}"; };
function echo_BIGre() { echo -e "${BIGre}$@${Reg}"; };
function echo_BIYel() { echo -e "${BIYel}$@${Reg}"; };
function echo_BIBlu() { echo -e "${BIBlu}$@${Reg}"; };
function echo_BIPur() { echo -e "${BIPur}$@${Reg}"; };
function echo_BICya() { echo -e "${BICya}$@${Reg}"; };
function echo_BIWhi() { echo -e "${BIWhi}$@${Reg}"; };
# Background
function echo_On_Bla() { echo -e "${On_Bla}$@${Reg}"; };
function echo_On_Red() { echo -e "${On_Red}$@${Reg}"; };
function echo_On_Gre() { echo -e "${On_Gre}$@${Reg}"; };
function echo_On_Yel() { echo -e "${On_Yel}$@${Reg}"; };
function echo_On_Blu() { echo -e "${On_Blu}$@${Reg}"; };
function echo_On_Pur() { echo -e "${On_Pur}$@${Reg}"; };
function echo_On_Cya() { echo -e "${On_Cya}$@${Reg}"; };
function echo_On_Whi() { echo -e "${On_Whi}$@${Reg}"; };
# High Intensity Backgrounds
function echo_On_IBla() { echo -e "${On_IBla}$@${Reg}"; };
function echo_On_IRed() { echo -e "${On_IRed}$@${Reg}"; };
function echo_On_IGre() { echo -e "${On_IGre}$@${Reg}"; };
function echo_On_IYel() { echo -e "${On_IYel}$@${Reg}"; };
function echo_On_IBlu() { echo -e "${On_IBlu}$@${Reg}"; };
function echo_On_IPur() { echo -e "${On_IPur}$@${Reg}"; };
function echo_On_ICya() { echo -e "${On_ICya}$@${Reg}"; };
function echo_On_IWhi() { echo -e "${On_IWhi}$@${Reg}"; };

# Based on answer https://unix.stackexchange.com/a/114129
function colorout() {
    color=$1;
    IFS='';
    while read data; do
	echo -e "$color$data${Reg}";
    done;
};
