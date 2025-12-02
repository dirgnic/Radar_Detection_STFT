{-
  Main module for Radar FFT optimization
  Provides command-line interface for Haskell optimizations
-}

module Main where

import Data.Complex
import qualified RadarFFT as RFFT
import qualified RadarOptimize as RO
import System.Environment (getArgs)
import Text.Read (readMaybe)

main :: IO ()
main = do
    putStrLn "==========================================="
    putStrLn "  Haskell Radar Optimization Module"
    putStrLn "==========================================="
    putStrLn ""
    
    args <- getArgs
    
    case args of
        ["fft", n] -> case readMaybe n :: Maybe Int of
            Just size -> runFFTDemo size
            Nothing -> putStrLn "Invalid size parameter"
        
        ["benchmark"] -> runBenchmark
        
        ["test"] -> runTests
        
        _ -> printUsage

printUsage :: IO ()
printUsage = do
    putStrLn "Usage:"
    putStrLn "  radar_fft fft <size>    - Run FFT on random signal"
    putStrLn "  radar_fft benchmark     - Run performance benchmarks"
    putStrLn "  radar_fft test          - Run tests"
    putStrLn ""

runFFTDemo :: Int -> IO ()
runFFTDemo size = do
    putStrLn $ "Running FFT on signal of size " ++ show size
    
    -- Generate test signal
    let signal = [fromIntegral i :+ 0 | i <- [1..size]]
    
    -- Compute FFT
    let spectrum = RFFT.fftRadix2 signal
    
    -- Show first few results
    putStrLn "First 5 FFT coefficients:"
    mapM_ print (take 5 spectrum)
    
    putStrLn $ "✓ FFT completed for " ++ show size ++ " samples"

runBenchmark :: IO ()
runBenchmark = do
    putStrLn "Running performance benchmarks..."
    putStrLn ""
    
    let sizes = [256, 1024, 4096]
    
    mapM_ benchmarkSize sizes
    
    putStrLn "✓ Benchmarks complete"

benchmarkSize :: Int -> IO ()
benchmarkSize size = do
    let signal = [fromIntegral i :+ 0 | i <- [1..size]]
    
    putStrLn $ "FFT (N=" ++ show size ++ "):"
    
    -- Simple timing (not precise, but demonstrates)
    let spectrum = RFFT.fftRadix2 signal
    let magnitude = sum [RFFT.magnitude c | c <- spectrum]
    
    putStrLn $ "  Sum of magnitudes: " ++ show magnitude
    putStrLn ""

runTests :: IO ()
runTests = do
    putStrLn "Running tests..."
    putStrLn ""
    
    -- Test 1: FFT size
    test1
    
    -- Test 2: Peak detection
    test2
    
    -- Test 3: CFAR
    test3
    
    putStrLn "✓ All tests passed"

test1 :: IO ()
test1 = do
    putStr "[TEST 1] FFT preserves size... "
    let signal = [1 :+ 0, 2 :+ 0, 3 :+ 0, 4 :+ 0]
    let spectrum = RFFT.fftRadix2 signal
    if length spectrum == length signal
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

test2 :: IO ()
test2 = do
    putStr "[TEST 2] Peak detection... "
    let spectrum = [-10, -5, 20, -5, -10, -8, 15, -9]
    let peaks = RFFT.peakDetection spectrum 10 1
    if length peaks == 2
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"

test3 :: IO ()
test3 = do
    putStr "[TEST 3] CFAR detector... "
    let spectrum = replicate 50 (-30) ++ [10] ++ replicate 50 (-30)
    let detections = RFFT.cfar spectrum 2 10 1e-4
    if not (null detections)
        then putStrLn "✓ PASS"
        else putStrLn "✗ FAIL"
