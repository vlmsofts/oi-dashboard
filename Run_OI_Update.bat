@echo off
title VLM - Open Interest Monitor Update
cd /d "C:\Users\Louis\OneDrive - VLM Commodities LTD\Desktop\Open interest dashboard"

echo.
echo ================================================
echo   VLM Commodities - Open Interest Monitor
echo   Cotton  ^|  Sugar  ^|  Coffee  ^|  Cocoa
echo ================================================
echo.

echo Fetching latest OI data from Bloomberg...
echo ------------------------------------------------
python oi_fetcher.py
if errorlevel 1 (
    echo.
    echo WARNING: Data fetch returned an error.
    echo Check that Bloomberg Terminal is open and logged in.
    echo Continuing to image generation with existing data...
)

echo.
echo Generating WhatsApp images...
echo ------------------------------------------------
python build_whatsapp_oi.py

echo.
echo ================================================
echo   Done. Press any key to close.
echo ================================================
pause >nul
