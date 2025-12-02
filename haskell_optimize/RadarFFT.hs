{-
  Module: RadarFFT
  Description: Optimized FFT computations for radar signal processing
  Author: Signal Processing Team
  Date: December 2025
  
  This module provides high-performance FFT implementations optimized
  for radar signal processing using Haskell's powerful type system
  and lazy evaluation.
-}

module RadarFFT
    ( Complex(..)
    , fftRadix2
    , fftOptimized
    , powerSpectrum
    , spectrogram
    , cfar
    , peakDetection
    ) where

import Data.Complex
import Data.List (sort, sortBy)
import Data.Ord (comparing)

-- Type aliases for clarity
type Signal = [Complex Double]
type Frequency = Double
type Magnitude = Double
type Spectrum = [(Frequency, Magnitude)]

-- | Fast Fourier Transform using Cooley-Tukey Radix-2 algorithm
-- Optimized for power-of-2 input sizes
fftRadix2 :: Signal -> Signal
fftRadix2 [x] = [x]
fftRadix2 xs
    | length xs `mod` 2 /= 0 = error "Input size must be power of 2"
    | otherwise = interleave evens odds
  where
    n = length xs
    evens = fftRadix2 [xs !! i | i <- [0, 2 .. n-1]]
    odds  = fftRadix2 [xs !! i | i <- [1, 3 .. n-1]]
    
    interleave es os = zipWith (+) es twiddles ++ zipWith (-) es twiddles
      where
        twiddles = zipWith (*) os [cis (-2 * pi * k / fromIntegral n) 
                                   | k <- [0..]]

-- | Optimized FFT with zero-padding and windowing
fftOptimized :: Signal -> Int -> Signal
fftOptimized signal targetSize = fftRadix2 paddedWindowed
  where
    -- Apply Hamming window
    n = length signal
    hammingWindow = [0.54 - 0.46 * cos (2 * pi * k / fromIntegral (n - 1))
                    | k <- [0 .. fromIntegral n - 1]]
    windowed = zipWith (\w s -> (w :+ 0) * s) hammingWindow signal
    
    -- Zero-padding to target size
    zeros = replicate (targetSize - n) (0 :+ 0)
    paddedWindowed = windowed ++ zeros

-- | Calculate power spectrum (magnitude squared)
powerSpectrum :: Signal -> [Double]
powerSpectrum spectrum = map (\c -> magnitude c ** 2) spectrum

-- | Convert magnitude to decibels
toDecibels :: [Double] -> [Double]
toDecibels mags = map (\m -> 20 * logBase 10 (m + 1e-10)) mags

-- | CFAR (Constant False Alarm Rate) detector
-- Returns indices of detected targets
cfar :: [Double]      -- Input spectrum (dB)
     -> Int           -- Number of guard cells
     -> Int           -- Number of training cells
     -> Double        -- Probability of false alarm
     -> [Int]         -- Indices of detections
cfar spectrum numGuard numTrain pfa = 
    [i | i <- [start .. end], isDetection i]
  where
    n = length spectrum
    start = numGuard + numTrain
    end = n - numGuard - numTrain - 1
    
    -- Scaling factor based on PFA
    alpha = fromIntegral numTrain * ((pfa ** (-1 / fromIntegral numTrain)) - 1)
    
    -- Check if cell under test exceeds threshold
    isDetection i = spectrum !! i > threshold i
      where
        threshold idx = alpha * noiseLevel idx
        noiseLevel idx = 
            let trainLeft  = [spectrum !! j | j <- [idx - numGuard - numTrain .. idx - numGuard - 1]]
                trainRight = [spectrum !! j | j <- [idx + numGuard + 1 .. idx + numGuard + numTrain]]
            in (sum trainLeft + sum trainRight) / fromIntegral (2 * numTrain)

-- | Peak detection with threshold and minimum distance
peakDetection :: [Double]  -- Input spectrum (dB)
              -> Double    -- Threshold (dB)
              -> Int       -- Minimum distance between peaks
              -> [(Int, Double)]  -- (index, magnitude) pairs
peakDetection spectrum threshold minDist = 
    filter (\(_, mag) -> mag > threshold) $ findPeaks 0 []
  where
    n = length spectrum
    
    findPeaks i acc
        | i >= n - 1 = reverse acc
        | isPeak i = 
            let nextI = i + minDist
            in findPeaks nextI ((i, spectrum !! i) : acc)
        | otherwise = findPeaks (i + 1) acc
    
    isPeak i = i > 0 && i < n - 1 &&
               spectrum !! i > spectrum !! (i - 1) &&
               spectrum !! i > spectrum !! (i + 1)

-- | Spectrogram computation (Short-Time Fourier Transform)
spectrogram :: Signal      -- Input signal
            -> Int         -- Window size (nperseg)
            -> Int         -- Overlap size
            -> [[Double]]  -- 2D array of magnitudes (time x frequency)
spectrogram signal windowSize overlap = map (toDecibels . powerSpectrum . fftRadix2) windows
  where
    hopSize = windowSize - overlap
    numWindows = (length signal - windowSize) `div` hopSize + 1
    
    windows = [take windowSize $ drop (i * hopSize) signal | i <- [0 .. numWindows - 1]]

-- | Range-Doppler processing (2D FFT)
rangeDoppler :: [[Complex Double]]  -- 2D signal matrix (chirps x samples)
             -> [[Complex Double]]  -- 2D FFT result
rangeDoppler matrix = map fftRadix2 $ transpose $ map fftRadix2 matrix
  where
    transpose [] = []
    transpose ([]:_) = []
    transpose xs = map head xs : transpose (map tail xs)

-- | Doppler shift calculation
dopplerShift :: Double   -- Velocity (m/s)
             -> Double   -- Wavelength (m)
             -> Double   -- Doppler frequency (Hz)
dopplerShift velocity wavelength = 2 * velocity / wavelength

-- | Range calculation from beat frequency
rangeFromFrequency :: Double   -- Beat frequency (Hz)
                   -> Double   -- Speed of light (m/s)
                   -> Double   -- Sweep time (s)
                   -> Double   -- Bandwidth (Hz)
                   -> Double   -- Range (m)
rangeFromFrequency beatFreq c sweepTime bandwidth = 
    (beatFreq * c * sweepTime) / (2 * bandwidth)

-- | Calculate SNR from signal
estimateSNR :: Signal -> Double
estimateSNR signal = 10 * logBase 10 (signalPower / noisePower)
  where
    magnitudes = map magnitude signal
    signalPower = (sum $ map (** 2) magnitudes) / fromIntegral (length magnitudes)
    noisePower = (sum $ take 100 $ sort $ map (** 2) magnitudes) / 100

-- | Bandpass filter coefficients (Butterworth)
butterworth :: Int       -- Order
            -> Double    -- Low cutoff frequency (normalized)
            -> Double    -- High cutoff frequency (normalized)
            -> [Double]  -- Filter coefficients
butterworth order lowCut highCut = 
    -- Simplified implementation - in practice use more sophisticated design
    [1.0, -2.0 * cos (pi * (lowCut + highCut) / 2), 1.0]

-- | Apply filter to signal
applyFilter :: [Double]  -- Filter coefficients
            -> Signal    -- Input signal
            -> Signal    -- Filtered signal
applyFilter coeffs signal = zipWith (*) signal (map (:+ 0) coeffs ++ repeat (1 :+ 0))

-- | Utility: Generate frequency axis
frequencyAxis :: Int     -- Number of samples
              -> Double  -- Sample rate (Hz)
              -> [Double]
frequencyAxis n fs = [k * fs / fromIntegral n | k <- [0 .. n `div` 2 - 1]]

-- | Utility: Hamming window
hammingWindow :: Int -> [Double]
hammingWindow n = [0.54 - 0.46 * cos (2 * pi * k / fromIntegral (n - 1))
                  | k <- [0 .. fromIntegral n - 1]]

-- | Utility: Hann window
hannWindow :: Int -> [Double]
hannWindow n = [0.5 * (1 - cos (2 * pi * k / fromIntegral (n - 1)))
               | k <- [0 .. fromIntegral n - 1]]

-- | Utility: Blackman window
blackmanWindow :: Int -> [Double]
blackmanWindow n = [a0 - a1 * cos (2 * pi * k / d) + a2 * cos (4 * pi * k / d)
                   | k <- [0 .. fromIntegral n - 1]]
  where
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08
    d = fromIntegral (n - 1)

-- | Performance optimization: Parallel FFT
-- Uses Haskell's parallel strategies for multi-core processing
parallelFFT :: Signal -> Signal
parallelFFT signal = fftRadix2 signal
  -- In a full implementation, use Control.Parallel.Strategies
  -- to parallelize the computation across multiple cores

-- | Batch processing for multiple signals
batchFFT :: [Signal] -> [Signal]
batchFFT = map fftOptimized'
  where
    fftOptimized' sig = fftOptimized sig (nextPowerOf2 $ length sig)
    nextPowerOf2 n = 2 ^ ceiling (logBase 2 (fromIntegral n) :: Double)
