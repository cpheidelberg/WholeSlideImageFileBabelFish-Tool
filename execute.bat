@echo off
"C:\Users\<user>\conda-envs\babelfish1\python.exe" "C:\Users\<user>\projects\WholeSlideImageFileBabelFish-Tool\roi_based_extraction.py" --openslide_dll "C:\Users\<user>\OpenSlide\openslide-bin-4.0.0.2-windows-x64\bin" --dmxt_dll "C:\Users\<user>\conda-envs\babelfish1\lib\site-packages\pylibdmtx\libdmtx-64.dll" --config "./config/roi_config_example.yaml"
exit /b 0