{-
  Module: RadarOptimize
  Description: High-performance optimization utilities for radar processing
  Author: Signal Processing Team
  
  Provides optimized algorithms for:
  - Target detection
  - Tracking
  - Signal correlation
  - Matrix operations
-}

module RadarOptimize
    ( Target(..)
    , detectTargets
    , trackTargets
    , correlation
    , matrixMultiply
    , kalmanFilter
    ) where

import Data.List (minimumBy, sortBy)
import Data.Ord (comparing)

-- | Target representation
data Target = Target
    { targetId :: Int
    , range :: Double      -- meters
    , velocity :: Double   -- m/s
    , snr :: Double        -- dB
    , timestamp :: Double  -- seconds
    } deriving (Show, Eq)

-- | Detection result with confidence
data Detection = Detection
    { detRange :: Double
    , detVelocity :: Double
    , detConfidence :: Double
    } deriving (Show)

-- | Track state for Kalman filter
data TrackState = TrackState
    { stateVector :: [Double]      -- [x, vx, y, vy]
    , covariance :: [[Double]]     -- State covariance matrix
    , processNoise :: [[Double]]   -- Process noise covariance
    , measureNoise :: [[Double]]   -- Measurement noise covariance
    } deriving (Show)

-- | Detect targets from spectrum peaks
detectTargets :: [(Int, Double)]  -- Peak indices and magnitudes
              -> Double            -- Sample rate
              -> Double            -- Chirp rate
              -> [Detection]
detectTargets peaks sampleRate chirpRate = 
    map peakToTarget peaks
  where
    peakToTarget (idx, mag) = Detection
        { detRange = rangeFromIndex idx sampleRate chirpRate
        , detVelocity = 0  -- Would need Doppler processing
        , detConfidence = sigmoid mag
        }
    
    rangeFromIndex i fs k = (fromIntegral i * 3e8 * 0.001) / (2 * 100e6)
    sigmoid x = 1 / (1 + exp (-x / 10))

-- | Track targets between frames
trackTargets :: [Target]    -- Previous frame targets
             -> [Detection] -- Current frame detections
             -> Double      -- Max association distance
             -> ([Target], [Detection])  -- (Matched, Unmatched)
trackTargets prevTargets currentDets maxDist = 
    foldl associate ([], currentDets) prevTargets
  where
    associate (matched, dets) target = 
        case findClosest target dets of
            Just (det, remaining) -> 
                let updated = updateTarget target det
                in (updated : matched, remaining)
            Nothing -> (matched, dets)
    
    findClosest _ [] = Nothing
    findClosest target dets = 
        let distances = map (\d -> (distance target d, d)) dets
            (minDist, closestDet) = minimumBy (comparing fst) distances
        in if minDist < maxDist
           then Just (closestDet, filter (/= closestDet) dets)
           else Nothing
    
    distance t d = sqrt ((range t - detRange d) ** 2 + 
                        (velocity t - detVelocity d) ** 2 / 100)
    
    updateTarget target det = target
        { range = detRange det
        , velocity = detVelocity det
        , timestamp = timestamp target + 0.001
        }

-- | Cross-correlation for signal matching
correlation :: [Double]  -- Signal 1
            -> [Double]  -- Signal 2
            -> [Double]  -- Correlation result
correlation sig1 sig2 = 
    [sum [s1 * s2' | (s1, s2') <- zip sig1 (drop lag sig2)]
    | lag <- [0 .. length sig2 - 1]]

-- | Auto-correlation
autoCorrelation :: [Double] -> [Double]
autoCorrelation sig = correlation sig sig

-- | Matrix multiplication (for Kalman filter)
matrixMultiply :: [[Double]] -> [[Double]] -> [[Double]]
matrixMultiply a b = 
    [[sum $ zipWith (*) rowA colB | colB <- transpose b] | rowA <- a]
  where
    transpose [] = []
    transpose ([]:_) = []
    transpose xs = map head xs : transpose (map tail xs)

-- | Matrix addition
matrixAdd :: [[Double]] -> [[Double]] -> [[Double]]
matrixAdd = zipWith (zipWith (+))

-- | Matrix subtraction
matrixSub :: [[Double]] -> [[Double]] -> [[Double]]
matrixSub = zipWith (zipWith (-))

-- | Scalar matrix multiplication
scalarMultiply :: Double -> [[Double]] -> [[Double]]
scalarMultiply s = map (map (* s))

-- | Kalman filter prediction step
kalmanPredict :: TrackState -> [[Double]] -> TrackState
kalmanPredict state transitionMatrix = TrackState
    { stateVector = matMulVec transitionMatrix (stateVector state)
    , covariance = matrixAdd 
        (matrixMultiply (matrixMultiply transitionMatrix (covariance state)) 
                       (transpose transitionMatrix))
        (processNoise state)
    , processNoise = processNoise state
    , measureNoise = measureNoise state
    }
  where
    matMulVec mat vec = map (sum . zipWith (*) vec) mat
    transpose m = [[m !! j !! i | j <- [0..length m - 1]] 
                  | i <- [0..length (head m) - 1]]

-- | Kalman filter update step
kalmanUpdate :: TrackState -> [Double] -> [[Double]] -> TrackState
kalmanUpdate state measurement measurementMatrix = TrackState
    { stateVector = zipWith (+) (stateVector state) 
                               (map (* kalmanGain) innovation)
    , covariance = scalarMultiply (1 - kalmanGain) (covariance state)
    , processNoise = processNoise state
    , measureNoise = measureNoise state
    }
  where
    innovation = zipWith (-) measurement 
                            (matMulVec measurementMatrix (stateVector state))
    kalmanGain = let s = head $ head $ matrixAdd
                           (matrixMultiply (matrixMultiply measurementMatrix 
                                                          (covariance state))
                                         (transpose measurementMatrix))
                           (measureNoise state)
                 in head (head (covariance state)) / s
    
    matMulVec mat vec = map (sum . zipWith (*) vec) mat
    transpose m = [[m !! j !! i | j <- [0..length m - 1]] 
                  | i <- [0..length (head m) - 1]]

-- | Kalman filter complete cycle
kalmanFilter :: TrackState -> [[Double]] -> [Double] -> [[Double]] -> TrackState
kalmanFilter state transitionMat measurement measureMat =
    kalmanUpdate (kalmanPredict state transitionMat) measurement measureMat

-- | Adaptive threshold calculation
adaptiveThreshold :: [Double]  -- Signal
                  -> Double    -- Percentile (0-1)
                  -> Double    -- Threshold value
adaptiveThreshold signal percentile = 
    sorted !! idx
  where
    sorted = sortBy compare signal
    idx = floor (percentile * fromIntegral (length signal))

-- | Moving average filter
movingAverage :: Int        -- Window size
              -> [Double]   -- Input signal
              -> [Double]   -- Smoothed signal
movingAverage windowSize signal = 
    [average (take windowSize $ drop i signal) 
    | i <- [0 .. length signal - windowSize]]
  where
    average xs = sum xs / fromIntegral (length xs)

-- | Exponential moving average
exponentialMovingAverage :: Double    -- Alpha (smoothing factor)
                         -> [Double]  -- Input signal
                         -> [Double]  -- Smoothed signal
exponentialMovingAverage alpha signal = 
    scanl (\acc x -> alpha * x + (1 - alpha) * acc) (head signal) (tail signal)

-- | Median filter for noise reduction
medianFilter :: Int        -- Window size
             -> [Double]   -- Input signal
             -> [Double]   -- Filtered signal
medianFilter windowSize signal = 
    [median (take windowSize $ drop i signal)
    | i <- [0 .. length signal - windowSize]]
  where
    median xs = let sorted = sortBy compare xs
                    n = length sorted
                in if odd n 
                   then sorted !! (n `div` 2)
                   else (sorted !! (n `div` 2 - 1) + sorted !! (n `div` 2)) / 2

-- | Signal-to-noise ratio estimation
estimateSNR :: [Double]  -- Signal with noise
            -> [Double]  -- Noise reference
            -> Double    -- SNR in dB
estimateSNR signal noise = 10 * logBase 10 (signalPower / noisePower)
  where
    signalPower = sum (map (** 2) signal) / fromIntegral (length signal)
    noisePower = sum (map (** 2) noise) / fromIntegral (length noise)

-- | Distance between two points (Euclidean)
euclideanDistance :: [Double] -> [Double] -> Double
euclideanDistance p1 p2 = sqrt $ sum $ zipWith (\x y -> (x - y) ** 2) p1 p2

-- | Mahalanobis distance (for covariance-aware matching)
mahalanobisDistance :: [Double]    -- Point 1
                    -> [Double]    -- Point 2
                    -> [[Double]]  -- Covariance matrix
                    -> Double
mahalanobisDistance p1 p2 cov = 
    sqrt $ sum $ zipWith (*) diff (matMulVec invCov diff)
  where
    diff = zipWith (-) p1 p2
    invCov = invertMatrix cov
    matMulVec mat vec = map (sum . zipWith (*) vec) mat
    
    -- Simplified matrix inversion (for 2x2)
    invertMatrix [[a, b], [c, d]] = 
        let det = a * d - b * c
        in [[d/det, -b/det], [-c/det, a/det]]
    invertMatrix _ = [[1, 0], [0, 1]]  -- Identity fallback

-- | Hungarian algorithm for optimal assignment
hungarianAssignment :: [[Double]]  -- Cost matrix
                    -> [(Int, Int)] -- (row, col) assignments
hungarianAssignment costs = 
    -- Simplified greedy assignment (full Hungarian is complex)
    assignGreedy costs [] 0
  where
    assignGreedy [] acc _ = reverse acc
    assignGreedy (row:rows) acc rowIdx = 
        let colIdx = minIndex row
        in assignGreedy rows ((rowIdx, colIdx) : acc) (rowIdx + 1)
    
    minIndex xs = snd $ minimumBy (comparing fst) $ zip xs [0..]
