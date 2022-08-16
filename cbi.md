# CBI data

About repository
* repo: https://github.com/jngkim/qmcpack.git
* p3hpc-cbi branch
* Removed AFQMC/formic/QMCTools and CUDA_legacy to correctly calculate total and common LOCs.

cmake Options to create compile_commands.json for each plaform
* common :  -DQMC_MIXED_PRECISION=ON -DQMC_MPI=OFF
* OMPT :   -DENABLE_OFFLOAD=ON  -DOFFLOAD_TARGET=spir64 
* OMPTcuda : -DENABLE_OFFLOAD=ON -DENABLE_CUDA=ON 
* OMPThip : -DENABLE_OFFLOAD=ON -DENABLE_CUDA=ON -DQMC_CUDA2HIP=ON
* OMPTsycl : -DENABLE_OFFLOAD=ON -DOFFLOAD_TARGET=spir64 -DENABLE_SYCL=ON 


## OMPT+COMPcuda+OMPTsycl
```
-------------------------------------------------
                       Platform Set    LOC % LOC
-------------------------------------------------
                                 {}  29071 16.11
                         {OMPTsycl}   1068  0.59
                          {OMPThip}    570  0.32
                         {OMPTcuda}   1629  0.90
                             {OMPT}      2  0.00
                {OMPTcuda, OMPThip}   2281  1.26
                   {OMPTsycl, OMPT}      6  0.00
          {OMPTcuda, OMPT, OMPThip}      6  0.00
      {OMPTcuda, OMPTsycl, OMPThip}    130  0.07
         {OMPTcuda, OMPTsycl, OMPT}      3  0.00
{OMPTcuda, OMPTsycl, OMPT, OMPThip} 145689 80.73
-------------------------------------------------
Code Divergence: 0.02
Unused Code (%): 16.11
Total SLOC: 180455

Distance Matrix
----------------------------------------
         OMPT OMPTcuda OMPThip OMPTsycl
----------------------------------------
    OMPT 0.00     0.03    0.02     0.01
OMPTcuda 0.03     0.00    0.01     0.03
 OMPThip 0.02     0.01    0.00     0.03
OMPTsycl 0.01     0.03    0.03     0.00
----------------------------------------

```

## CPU+OMPT+COMPcuda+OMPTsycl

```
qmcpack.yaml:
codebase:
  files: [ src/**/*.cpp, src/**/*.hpp, src/**/*.cu, src/**/*.h, src/**/*.c]
  exclude_files: [src/**/tests/*]
  platforms: [ CPU, OMPT, OMPTcuda, OMPTsycl]

CPU:
  commands: build_cpu/compile_commands.json

OMPT:
  commands: build_ompt/compile_commands.json

OMPTcuda:
  commands: build_ompt_cuda/compile_commands.json

OMPTsycl:
  commands: build_ompt_sycl/compile_commands.json


---------------------------------------------
                   Platform Set    LOC % LOC
---------------------------------------------
                             {}  29628 16.42
                     {OMPTcuda}   3910  2.17
                     {OMPTsycl}   1068  0.59
                          {CPU}     13  0.01
               {OMPTsycl, OMPT}      1  0.00
               {OMPTcuda, OMPT}      1  0.00
           {OMPTsycl, OMPTcuda}    130  0.07
                    {CPU, OMPT}      2  0.00
     {OMPTcuda, OMPTsycl, OMPT}    516  0.29
          {CPU, OMPTsycl, OMPT}      5  0.00
          {CPU, OMPTcuda, OMPT}      5  0.00
{CPU, OMPTcuda, OMPTsycl, OMPT} 145176 80.45
---------------------------------------------
Code Divergence: 0.02
Unused Code (%): 16.42
Total SLOC: 180455

Distance Matrix
-------------------------------------
          CPU OMPT OMPTcuda OMPTsycl
-------------------------------------
     CPU 0.00 0.00     0.03     0.01
    OMPT 0.00 0.00     0.03     0.01
OMPTcuda 0.03 0.03     0.00     0.03
OMPTsycl 0.01 0.01     0.03     0.00
-------------------------------------
```


# Add OMPThip

```
------------------------------------------------------
                            Platform Set    LOC % LOC
------------------------------------------------------
                                      {}  29058 16.10
                              {OMPTsycl}   1068  0.59
                                   {CPU}     13  0.01
                               {OMPThip}    570  0.32
                              {OMPTcuda}   1629  0.90
                     {OMPTcuda, OMPThip}   2281  1.26
                        {OMPT, OMPTsycl}      1  0.00
                             {OMPT, CPU}      2  0.00
                   {OMPT, OMPTsycl, CPU}      5  0.00
               {OMPThip, OMPT, OMPTcuda}      1  0.00
           {OMPTcuda, OMPThip, OMPTsycl}    130  0.07
     {OMPThip, OMPT, OMPTcuda, OMPTsycl}    516  0.29
          {OMPThip, OMPT, OMPTcuda, CPU}      5  0.00
         {OMPT, OMPTcuda, OMPTsycl, CPU}      3  0.00
{OMPT, CPU, OMPTcuda, OMPThip, OMPTsycl} 145173 80.45
------------------------------------------------------
Code Divergence: 0.02
Unused Code (%): 16.10
Total SLOC: 180455

Distance Matrix
---------------------------------------------
          CPU OMPT OMPTcuda OMPThip OMPTsycl
---------------------------------------------
     CPU 0.00 0.00     0.03    0.02     0.01
    OMPT 0.00 0.00     0.03    0.02     0.01
OMPTcuda 0.03 0.03     0.00    0.01     0.03
 OMPThip 0.02 0.02     0.01    0.00     0.03
OMPTsycl 0.01 0.01     0.03    0.03     0.00
---------------------------------------------
```

## CPU w/wo MPI
```
--------------------------
Platform Set    LOC % LOC
--------------------------
          {}  33281 18.44
       {MPI}   1973  1.09
       {CPU}    214  0.12
  {MPI, CPU} 144987 80.35
--------------------------
Code Divergence: 0.01
Unused Code (%): 18.44
Total SLOC: 180455

Distance Matrix
--------------
     CPU  MPI
--------------
CPU 0.00 0.01
MPI 0.01 0.00
--------------
```

Unused 29058 contains only 1973 MPI codes. 

## CUDA_legacy removed
```
---------------------------------------------
                   Platform Set    LOC % LOC
---------------------------------------------
                             {}  33949 16.17
                     {OMPTcuda}   4834  2.30
                     {OMPTsycl}   1122  0.53
                          {CPU}    263  0.13
           {OMPTcuda, OMPTsycl}    132  0.06
               {OMPT, OMPTsycl}      3  0.00
               {OMPT, OMPTcuda}      3  0.00
                    {OMPT, CPU}      2  0.00
     {OMPT, OMPTcuda, OMPTsycl}    639  0.30
          {OMPT, CPU, OMPTsycl}    331  0.16
          {OMPT, CPU, OMPTcuda}      5  0.00
{OMPT, CPU, OMPTcuda, OMPTsycl} 168622 80.33
---------------------------------------------
Code Divergence: 0.02
Unused Code (%): 16.17
Total SLOC: 209905

Distance Matrix
-------------------------------------
          CPU OMPT OMPTcuda OMPTsycl
-------------------------------------
     CPU 0.00 0.01     0.04     0.01
    OMPT 0.01 0.00     0.03     0.01
OMPTcuda 0.04 0.03     0.00     0.04
OMPTsycl 0.01 0.01     0.04     0.00
-------------------------------------
```


## With CUDA_legacy
```
---------------------------------------------
                   Platform Set    LOC % LOC
---------------------------------------------
                             {}  50143 22.18
                     {OMPTcuda}   4834  2.14
                     {OMPTsycl}   1122  0.50
                          {CPU}    263  0.12
           {OMPTcuda, OMPTsycl}    132  0.06
               {OMPTsycl, OMPT}      3  0.00
               {OMPTcuda, OMPT}      3  0.00
                    {CPU, OMPT}      2  0.00
     {OMPTcuda, OMPT, OMPTsycl}    639  0.28
          {OMPTsycl, CPU, OMPT}    331  0.15
          {OMPTcuda, CPU, OMPT}      5  0.00
{OMPTcuda, CPU, OMPT, OMPTsycl} 168622 74.58
---------------------------------------------
Code Divergence: 0.02
Unused Code (%): 22.18
Total SLOC: 226099

Distance Matrix
-------------------------------------
          CPU OMPT OMPTcuda OMPTsycl
-------------------------------------
     CPU 0.00 0.01     0.04     0.01
    OMPT 0.01 0.00     0.03     0.01
OMPTcuda 0.04 0.03     0.00     0.04
OMPTsycl 0.01 0.01     0.04     0.00
-------------------------------------

```

## code-base investigator

https://github.com/intel/code-base-investigator.git 

Workaround to apply CBI to QMCPACK
* `*.def` are sycl macro files.
```
$ git diff
diff --git a/codebasin/language.py b/codebasin/language.py
index 079fcbb..b6517f4 100644
--- a/codebasin/language.py
+++ b/codebasin/language.py
@@ -26,7 +26,7 @@ class FileLanguage:
     _language_extensions['c++'] = ['.c++', '.cxx', '.cpp', '.cc',
                                    '.hpp', '.hxx', '.h++', '.hh',
                                    '.inc', '.inl', '.tcc', '.icc',
-                                   '.ipp', '.cu', '.cuh', '.cl']
+                                   '.ipp', '.cu', '.cuh', '.cl' ,'.def']
     _language_extensions['asm'] = ['.s', '.S', '.asm']

     def __init__(self, filename):
```
* Parsing boost can be taxing
```
diff --git a/codebasin/preprocessor.py b/codebasin/preprocessor.py
index f63d5dc..85b01da 100644
--- a/codebasin/preprocessor.py
+++ b/codebasin/preprocessor.py
@@ -1998,10 +1998,11 @@ class ExpressionEvaluator(Parser):
             # Convert to decimal and then to integer with correct sign
             # Preprocessor always uses 64-bit arithmetic!
             int_value = int(value, base)
-            if suffix and 'u' in suffix:
-                return np.uint64(int_value)
-            else:
-                return np.int64(int_value)
+            #if suffix and 'u' in suffix:
+            #    return np.uint64(int_value)
+            #else:
+            #    return np.int64(int_value)
+            return int_value
         except ParseError:
             self.pos = initial_pos
```

