import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, ChevronLeft } from 'lucide-react';

const CNNVisualization = () => {
  // MNIST примерни данни (опростени 14x14 за по-добра визуализация)
  const mnistSamples = [
    { label: 0, data: [[0,0,0,0,1,1,1,1,1,1,0,0,0,0],[0,0,0,1,1,1,1,1,1,1,1,0,0,0],[0,0,1,1,1,0,0,0,0,1,1,1,0,0],[0,1,1,1,0,0,0,0,0,0,1,1,1,0],[0,1,1,0,0,0,0,0,0,0,1,1,1,0],[1,1,1,0,0,0,0,0,0,0,0,1,1,1],[1,1,1,0,0,0,0,0,0,0,0,1,1,1],[1,1,1,0,0,0,0,0,0,0,0,1,1,1],[1,1,1,0,0,0,0,0,0,0,0,1,1,1],[0,1,1,1,0,0,0,0,0,0,1,1,1,0],[0,1,1,1,0,0,0,0,0,0,1,1,1,0],[0,0,1,1,1,0,0,0,0,1,1,1,0,0],[0,0,0,1,1,1,1,1,1,1,1,0,0,0],[0,0,0,0,1,1,1,1,1,1,0,0,0,0]] },
    { label: 1, data: [[0,0,0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,1,1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,1,1,0,0,0,0,0,0],[0,0,1,1,1,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,0,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,0,0,0,0]] },
    { label: 7, data: [[1,1,1,1,1,1,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1,1,1,1,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,1,1,1,0,0,0,0,0],[0,0,0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0,0,0]] }
  ];

  const [currentSample, setCurrentSample] = useState(0);
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);
  const [poolingSize, setPoolingSize] = useState(2);
  const [kernelSize, setKernelSize] = useState(3);
  const [currentFilter, setCurrentFilter] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationPos, setAnimationPos] = useState({ x: 0, y: 0 });
  const [showConvolution, setShowConvolution] = useState(true);
  const [showPooling, setShowPooling] = useState(false);
  const [poolingPos, setPoolingPos] = useState({ x: 0, y: 0 });
  const [isPoolingAnimating, setIsPoolingAnimating] = useState(false);

  // Примерни филтри за детекция на ръбове
  const filters = [
    { name: 'Хоризонтални ръбове', kernel: [[-1,-1,-1],[0,0,0],[1,1,1]] },
    { name: 'Вертикални ръбове', kernel: [[-1,0,1],[-1,0,1],[-1,0,1]] },
    { name: 'Диагонални ръбове', kernel: [[0,1,1],[-1,0,1],[-1,-1,0]] }
  ];

  // Padding функция
  const padArray = (arr, padding) => {
    if (padding === 0) return arr;
    const size = arr.length;
    const newSize = size + 2 * padding;
    const padded = Array(newSize).fill(null).map(() => Array(newSize).fill(0));
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        padded[i + padding][j + padding] = arr[i][j];
      }
    }
    return padded;
  };

  // Конволюционна операция
  const convolve = (input, kernel, stride, padding) => {
    const inputSize = input.length;
    const kernelSize = kernel.length;
    const paddedInput = padArray(input, padding);
    const outputSize = Math.floor((paddedInput.length - kernelSize) / stride) + 1;
    const output = Array(outputSize).fill(null).map(() => Array(outputSize).fill(0));

    for (let i = 0; i < outputSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        let sum = 0;
        for (let ki = 0; ki < kernelSize; ki++) {
          for (let kj = 0; kj < kernelSize; kj++) {
            const inputRow = i * stride + ki;
            const inputCol = j * stride + kj;
            sum += paddedInput[inputRow][inputCol] * kernel[ki][kj];
          }
        }
        output[i][j] = Math.max(0, sum); // ReLU activation
      }
    }
    return output;
  };

  // Изчисляване на текуща конволюция
  const calculateCurrentConvolution = () => {
    const inputData = mnistSamples[currentSample].data;
    const paddedInput = padArray(inputData, padding);
    const kernel = filters[currentFilter].kernel;
    const { x, y } = animationPos;
    
    let sum = 0;
    const details = [];
    
    for (let ki = 0; ki < kernelSize; ki++) {
      for (let kj = 0; kj < kernelSize; kj++) {
        const inputRow = y * stride + ki;
        const inputCol = x * stride + kj;
        const inputVal = paddedInput[inputRow][inputCol];
        const kernelVal = kernel[ki][kj];
        const product = inputVal * kernelVal;
        sum += product;
        details.push({ inputVal, kernelVal, product });
      }
    }
    
    const output = Math.max(0, sum); // ReLU
    return { sum, output, details };
  };

  // Изчисляване на текущ pooling
  const calculateCurrentPooling = () => {
    const fm = featureMaps[currentFilter];
    const { x, y } = poolingPos;
    
    const values = [];
    let max = -Infinity;
    
    for (let pi = 0; pi < poolingSize; pi++) {
      for (let pj = 0; pj < poolingSize; pj++) {
        const row = y * poolingSize + pi;
        const col = x * poolingSize + pj;
        if (row < fm.length && col < fm[0].length) {
          const val = fm[row][col];
          values.push({ row, col, val });
          if (val > max) max = val;
        }
      }
    }
    
    return { values, max };
  };

  // Max Pooling
  const maxPool = (input, poolSize) => {
    const inputSize = input.length;
    const outputSize = Math.floor(inputSize / poolSize);
    const output = Array(outputSize).fill(null).map(() => Array(outputSize).fill(0));

    for (let i = 0; i < outputSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        let max = -Infinity;
        for (let pi = 0; pi < poolSize; pi++) {
          for (let pj = 0; pj < poolSize; pj++) {
            const val = input[i * poolSize + pi][j * poolSize + pj];
            if (val > max) max = val;
          }
        }
        output[i][j] = max;
      }
    }
    return output;
  };

  // Изчисляване на feature maps
  const inputData = mnistSamples[currentSample].data;
  const paddedInput = padArray(inputData, padding);
  const featureMaps = filters.map(f => convolve(inputData, f.kernel, stride, padding));
  const pooledMaps = featureMaps.map(fm => maxPool(fm, poolingSize));
  
  const maxOutputX = Math.floor((paddedInput.length - kernelSize) / stride);
  const maxOutputY = Math.floor((paddedInput.length - kernelSize) / stride);
  
  const maxPoolingX = Math.floor(featureMaps[0][0].length / poolingSize) - 1;
  const maxPoolingY = Math.floor(featureMaps[0].length / poolingSize) - 1;

  // Анимация на конволюция
  useEffect(() => {
    if (!isAnimating) return;
    const interval = setInterval(() => {
      setAnimationPos(prev => {
        if (prev.x >= maxOutputX) {
          if (prev.y >= maxOutputY) {
            setIsAnimating(false);
            return { x: 0, y: 0 };
          }
          return { x: 0, y: prev.y + 1 };
        }
        return { x: prev.x + 1, y: prev.y };
      });
    }, 500);
    return () => clearInterval(interval);
  }, [isAnimating, maxOutputX, maxOutputY]);

  // Анимация на pooling
  useEffect(() => {
    if (!isPoolingAnimating) return;
    const interval = setInterval(() => {
      setPoolingPos(prev => {
        if (prev.x >= maxPoolingX) {
          if (prev.y >= maxPoolingY) {
            setIsPoolingAnimating(false);
            return { x: 0, y: 0 };
          }
          return { x: 0, y: prev.y + 1 };
        }
        return { x: prev.x + 1, y: prev.y };
      });
    }, 500);
    return () => clearInterval(interval);
  }, [isPoolingAnimating, maxPoolingX, maxPoolingY]);

  // Стъпка напред/назад за конволюция
  const stepForward = () => {
    setAnimationPos(prev => {
      if (prev.x >= maxOutputX) {
        if (prev.y >= maxOutputY) return prev;
        return { x: 0, y: prev.y + 1 };
      }
      return { x: prev.x + 1, y: prev.y };
    });
  };

  const stepBackward = () => {
    setAnimationPos(prev => {
      if (prev.x === 0) {
        if (prev.y === 0) return prev;
        return { x: maxOutputX, y: prev.y - 1 };
      }
      return { x: prev.x - 1, y: prev.y };
    });
  };

  // Стъпка напред/назад за pooling
  const stepPoolingForward = () => {
    setPoolingPos(prev => {
      if (prev.x >= maxPoolingX) {
        if (prev.y >= maxPoolingY) return prev;
        return { x: 0, y: prev.y + 1 };
      }
      return { x: prev.x + 1, y: prev.y };
    });
  };

  const stepPoolingBackward = () => {
    setPoolingPos(prev => {
      if (prev.x === 0) {
        if (prev.y === 0) return prev;
        return { x: maxPoolingX, y: prev.y - 1 };
      }
      return { x: prev.x - 1, y: prev.y };
    });
  };

  const convResult = showConvolution ? calculateCurrentConvolution() : null;
  const poolingResult = showPooling ? calculateCurrentPooling() : null;

  // Рендериране на входно изображение с текуща позиция на кернела
  const renderInputWithKernel = () => {
    const { x, y } = animationPos;
    return (
      <div className="inline-block">
        {paddedInput.map((row, i) => (
          <div key={i} className="flex">
            {row.map((val, j) => {
              const isInKernel = showConvolution && 
                i >= y * stride && i < y * stride + kernelSize &&
                j >= x * stride && j < x * stride + kernelSize;
              
              const intensity = val * 255;
              const bgColor = `rgb(${255-intensity}, ${255-intensity}, 255)`;
              
              return (
                <div
                  key={j}
                  className={`border flex items-center justify-center transition-all ${
                    isInKernel ? 'border-red-500 border-2 bg-red-100' : 'border-gray-300'
                  }`}
                  style={{
                    width: '22px',
                    height: '22px',
                    backgroundColor: isInKernel ? undefined : bgColor
                  }}
                >
                  <span className="text-xs font-medium">{val}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    );
  };

  // Рендериране на кернела
  const renderKernel = () => {
    const kernel = filters[currentFilter].kernel;
    return (
      <div className="inline-block">
        {kernel.map((row, i) => (
          <div key={i} className="flex">
            {row.map((val, j) => {
              const bgColor = val >= 0 ? 'bg-blue-100' : 'bg-red-100';
              return (
                <div
                  key={j}
                  className={`border-2 border-gray-600 flex items-center justify-center ${bgColor}`}
                  style={{ width: '40px', height: '40px' }}
                >
                  <span className="text-sm font-bold">{val}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    );
  };

  // Рендериране на частичен feature map
  const renderPartialFeatureMap = () => {
    const fm = featureMaps[currentFilter];
    const { x: currentX, y: currentY } = animationPos;
    
    return (
      <div className="inline-block">
        {fm.map((row, i) => (
          <div key={i} className="flex">
            {row.map((val, j) => {
              const isCalculated = i < currentY || (i === currentY && j <= currentX);
              const isCurrent = i === currentY && j === currentX;
              
              const normalized = Math.abs(val) / 10;
              const intensity = Math.floor(Math.min(normalized, 1) * 255);
              const bgColor = isCalculated 
                ? `rgb(${255-intensity}, ${255-intensity}, 255)` 
                : 'rgb(240, 240, 240)';
              
              return (
                <div
                  key={j}
                  className={`border flex items-center justify-center transition-all ${
                    isCurrent ? 'border-green-500 border-2 ring-2 ring-green-300' : 'border-gray-300'
                  }`}
                  style={{
                    width: '28px',
                    height: '28px',
                    backgroundColor: bgColor
                  }}
                >
                  {isCalculated && <span className="text-xs font-medium">{val.toFixed(1)}</span>}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    );
  };

  // Рендериране на топлинна карта
  const renderHeatmap = (data, size = 20, showValues = false) => {
    const max = Math.max(...data.flat().map(v => Math.abs(v)));
    return (
      <div className="inline-block">
        {data.map((row, i) => (
          <div key={i} className="flex">
            {row.map((val, j) => {
              const normalized = max > 0 ? Math.abs(val) / max : 0;
              const intensity = Math.floor(normalized * 255);
              const bgColor = val >= 0 ? `rgb(${255-intensity}, ${255-intensity}, 255)` : `rgb(255, ${255-intensity}, ${255-intensity})`;
              return (
                <div
                  key={j}
                  className="border border-gray-300 flex items-center justify-center"
                  style={{
                    width: `${size}px`,
                    height: `${size}px`,
                    backgroundColor: bgColor,
                    fontSize: showValues ? '10px' : undefined
                  }}
                >
                  {showValues && <span className="font-medium">{val.toFixed(1)}</span>}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    );
  };

  // Рендериране на feature map с pooling рамка
  const renderFeatureMapWithPooling = () => {
    const fm = featureMaps[currentFilter];
    const { x: px, y: py } = poolingPos;
    
    return (
      <div className="inline-block">
        {fm.map((row, i) => (
          <div key={i} className="flex">
            {row.map((val, j) => {
              const isInPooling = showPooling && 
                i >= py * poolingSize && i < py * poolingSize + poolingSize &&
                j >= px * poolingSize && j < px * poolingSize + poolingSize;
              
              const normalized = Math.abs(val) / 10;
              const intensity = Math.floor(Math.min(normalized, 1) * 255);
              const bgColor = `rgb(${255-intensity}, ${255-intensity}, 255)`;
              
              return (
                <div
                  key={j}
                  className={`border flex items-center justify-center transition-all ${
                    isInPooling ? 'border-orange-500 border-2 bg-orange-100' : 'border-gray-300'
                  }`}
                  style={{
                    width: '26px',
                    height: '26px',
                    backgroundColor: isInPooling ? undefined : bgColor,
                    fontSize: '9px'
                  }}
                >
                  <span className="font-medium">{val.toFixed(1)}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    );
  };

  // Рендериране на частичен pooled map
  const renderPartialPooledMap = () => {
    const pm = pooledMaps[currentFilter];
    const { x: currentX, y: currentY } = poolingPos;
    
    return (
      <div className="inline-block">
        {pm.map((row, i) => (
          <div key={i} className="flex">
            {row.map((val, j) => {
              const isCalculated = i < currentY || (i === currentY && j <= currentX);
              const isCurrent = i === currentY && j === currentX;
              
              const normalized = Math.abs(val) / 10;
              const intensity = Math.floor(Math.min(normalized, 1) * 255);
              const bgColor = isCalculated 
                ? `rgb(${255-intensity}, ${255-intensity}, 255)` 
                : 'rgb(240, 240, 240)';
              
              return (
                <div
                  key={j}
                  className={`border flex items-center justify-center transition-all ${
                    isCurrent ? 'border-green-500 border-2 ring-2 ring-green-300' : 'border-gray-300'
                  }`}
                  style={{
                    width: '36px',
                    height: '36px',
                    backgroundColor: bgColor,
                    fontSize: '11px'
                  }}
                >
                  {isCalculated && <span className="font-bold">{val.toFixed(1)}</span>}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="w-full h-full bg-gradient-to-br from-blue-50 to-indigo-50 p-6 overflow-auto">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-2 text-indigo-900">
          Интерактивна визуализация на конволюционна невронна мрежа (CNN)
        </h1>
        <p className="text-center text-gray-600 mb-6">MNIST Dataset - Разпознаване на ръкописни цифри</p>

        {/* Контроли */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 text-indigo-800">Контролен панел</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Примерна цифра</label>
              <select 
                value={currentSample} 
                onChange={(e) => { setCurrentSample(Number(e.target.value)); setAnimationPos({ x: 0, y: 0 }); }}
                className="w-full p-2 border rounded"
              >
                {mnistSamples.map((s, i) => (
                  <option key={i} value={i}>Цифра {s.label}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Филтър</label>
              <select 
                value={currentFilter} 
                onChange={(e) => { setCurrentFilter(Number(e.target.value)); setAnimationPos({ x: 0, y: 0 }); }}
                className="w-full p-2 border rounded"
              >
                {filters.map((f, i) => (
                  <option key={i} value={i}>{f.name}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Stride (стъпка): {stride}</label>
              <input 
                type="range" 
                min="1" 
                max="2" 
                value={stride} 
                onChange={(e) => { setStride(Number(e.target.value)); setAnimationPos({ x: 0, y: 0 }); }}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Padding: {padding}</label>
              <input 
                type="range" 
                min="0" 
                max="2" 
                value={padding} 
                onChange={(e) => { setPadding(Number(e.target.value)); setAnimationPos({ x: 0, y: 0 }); }}
                className="w-full"
              />
            </div>
          </div>

          <div className="flex gap-2 mt-4 flex-wrap">
            <button
              onClick={stepBackward}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 flex items-center gap-2"
              disabled={animationPos.x === 0 && animationPos.y === 0}
            >
              <ChevronLeft size={16} />
              Назад
            </button>
            <button
              onClick={() => setIsAnimating(!isAnimating)}
              className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 flex items-center gap-2"
            >
              {isAnimating ? <Pause size={16} /> : <Play size={16} />}
              {isAnimating ? 'Пауза' : 'Авто'}
            </button>
            <button
              onClick={stepForward}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 flex items-center gap-2"
              disabled={animationPos.x >= maxOutputX && animationPos.y >= maxOutputY}
            >
              Напред
              <ChevronRight size={16} />
            </button>
            <button
              onClick={() => { setAnimationPos({ x: 0, y: 0 }); setIsAnimating(false); }}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 flex items-center gap-2"
            >
              <RotateCcw size={16} />
              Рестарт
            </button>
            <button
              onClick={() => setShowConvolution(!showConvolution)}
              className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
            >
              {showConvolution ? 'Скрий' : 'Покажи'} конволюция
            </button>
            <button
              onClick={() => { setShowPooling(!showPooling); setPoolingPos({ x: 0, y: 0 }); }}
              className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700"
            >
              {showPooling ? 'Скрий' : 'Покажи'} pooling
            </button>
          </div>
        </div>

        {/* Анимация на конволюцията */}
        {showConvolution && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4 text-indigo-800">
              🎬 Анимация на конволюцията - Позиция ({animationPos.x}, {animationPos.y})
            </h3>
            
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Входно изображение */}
              <div>
                <h4 className="font-semibold mb-3 text-center">Входно изображение</h4>
                <div className="flex justify-center">
                  {renderInputWithKernel()}
                </div>
                <p className="text-sm text-gray-600 mt-2 text-center">
                  Червена рамка = текуща позиция на кернела
                </p>
              </div>

              {/* Кернел и изчисление */}
              <div>
                <h4 className="font-semibold mb-3 text-center">{filters[currentFilter].name}</h4>
                <div className="flex justify-center mb-4">
                  {renderKernel()}
                </div>
                
                {convResult && (
                  <div className="bg-blue-50 p-4 rounded text-sm">
                    <p className="font-semibold mb-2">Изчисление:</p>
                    <div className="space-y-1 mb-2 max-h-32 overflow-y-auto">
                      {convResult.details.map((d, i) => (
                        <div key={i} className="text-xs">
                          {d.inputVal} × {d.kernelVal} = {d.product.toFixed(2)}
                        </div>
                      ))}
                    </div>
                    <div className="border-t pt-2">
                      <p><strong>Сума:</strong> {convResult.sum.toFixed(2)}</p>
                      <p><strong>ReLU:</strong> max(0, {convResult.sum.toFixed(2)}) = <span className="text-green-600 font-bold">{convResult.output.toFixed(2)}</span></p>
                    </div>
                  </div>
                )}
              </div>

              {/* Feature Map */}
              <div>
                <h4 className="font-semibold mb-3 text-center">Feature Map (в процес)</h4>
                <div className="flex justify-center">
                  {renderPartialFeatureMap()}
                </div>
                <p className="text-sm text-gray-600 mt-2 text-center">
                  Зелена рамка = току-що изчислена стойност
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Всички Feature Maps */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-indigo-800">
            📊 Завършени Feature Maps (след конволюция)
          </h3>
          <div className="flex gap-6 overflow-x-auto pb-4">
            {featureMaps.map((fm, idx) => (
              <div key={idx} className="flex-shrink-0">
                <p className="text-sm font-medium mb-2">{filters[idx].name}</p>
                {renderHeatmap(fm, 24, true)}
                <p className="text-xs text-gray-600 mt-1">
                  Размер: {fm.length}x{fm[0].length}
                </p>
              </div>
            ))}
          </div>
          <p className="text-sm text-gray-600 mt-4">
            ℹ️ Числата показват активацията след конволюция и ReLU. По-високи стойности = по-силна детекция на характеристика.
          </p>
        </div>

        {/* Анимация на Pooling */}
        {showPooling && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4 text-orange-800">
              🎬 Анимация на Max Pooling - Позиция ({poolingPos.x}, {poolingPos.y})
            </h3>
            
            <div className="flex gap-2 mb-4 flex-wrap">
              <button
                onClick={stepPoolingBackward}
                className="px-3 py-1 bg-gray-600 text-white rounded hover:bg-gray-700 flex items-center gap-1 text-sm"
                disabled={poolingPos.x === 0 && poolingPos.y === 0}
              >
                <ChevronLeft size={14} />
                Назад
              </button>
              <button
                onClick={() => setIsPoolingAnimating(!isPoolingAnimating)}
                className="px-3 py-1 bg-orange-600 text-white rounded hover:bg-orange-700 flex items-center gap-1 text-sm"
              >
                {isPoolingAnimating ? <Pause size={14} /> : <Play size={14} />}
                {isPoolingAnimating ? 'Пауза' : 'Авто'}
              </button>
              <button
                onClick={stepPoolingForward}
                className="px-3 py-1 bg-gray-600 text-white rounded hover:bg-gray-700 flex items-center gap-1 text-sm"
                disabled={poolingPos.x >= maxPoolingX && poolingPos.y >= maxPoolingY}
              >
                Напред
                <ChevronRight size={14} />
              </button>
              <button
                onClick={() => { setPoolingPos({ x: 0, y: 0 }); setIsPoolingAnimating(false); }}
                className="px-3 py-1 bg-gray-600 text-white rounded hover:bg-gray-700 flex items-center gap-1 text-sm"
              >
                <RotateCcw size={14} />
                Рестарт
              </button>
            </div>
            
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Feature Map с pooling рамка */}
              <div>
                <h4 className="font-semibold mb-3 text-center">Feature Map ({filters[currentFilter].name})</h4>
                <div className="flex justify-center">
                  {renderFeatureMapWithPooling()}
                </div>
                <p className="text-sm text-gray-600 mt-2 text-center">
                  Оранжева рамка = текуща pooling област ({poolingSize}x{poolingSize})
                </p>
              </div>

              {/* Max Pooling изчисление */}
              <div>
                <h4 className="font-semibold mb-3 text-center">Max Pooling операция</h4>
                
                {poolingResult && (
                  <div className="bg-orange-50 p-4 rounded">
                    <p className="font-semibold mb-2 text-sm">Стойности в текущата област:</p>
                    <div className="grid grid-cols-2 gap-2 mb-3">
                      {poolingResult.values.map((v, i) => (
                        <div 
                          key={i} 
                          className={`text-center p-2 rounded ${
                            v.val === poolingResult.max ? 'bg-green-200 font-bold' : 'bg-white'
                          }`}
                        >
                          {v.val.toFixed(2)}
                        </div>
                      ))}
                    </div>
                    <div className="border-t pt-3">
                      <p className="text-sm mb-1">Max Pooling избира максималната стойност:</p>
                      <p className="text-center">
                        <span className="text-2xl font-bold text-green-600">{poolingResult.max.toFixed(2)}</span>
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Pooled Map */}
              <div>
                <h4 className="font-semibold mb-3 text-center">Pooled Map (в процес)</h4>
                <div className="flex justify-center">
                  {renderPartialPooledMap()}
                </div>
                <p className="text-sm text-gray-600 mt-2 text-center">
                  Зелена рамка = току-що изчислена стойност
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Pooling слой - финални резултати */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-indigo-800">
            🔽 Финални Pooled Maps (Max Pooling {poolingSize}x{poolingSize})
          </h3>
          <div className="flex gap-6 overflow-x-auto pb-4">
            {pooledMaps.map((pm, idx) => (
              <div key={idx} className="flex-shrink-0">
                <p className="text-sm font-medium mb-2">{filters[idx].name}</p>
                {renderHeatmap(pm, 28, true)}
                <p className="text-xs text-gray-600 mt-1">
                  Размер: {pm.length}x{pm[0].length}
                </p>
              </div>
            ))}
          </div>
          <p className="text-sm text-gray-600 mt-4">
            ℹ️ Pooling намалява размерността {featureMaps[0].length}x{featureMaps[0][0].length} → {pooledMaps[0].length}x{pooledMaps[0][0].length} и прави мрежата по-устойчива на малки измествания.
          </p>
        </div>

        {/* Обяснения */}
        <div className="bg-gradient-to-r from-indigo-100 to-purple-100 rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4 text-indigo-900">
            📚 Как работи CNN?
          </h3>
          <div className="grid md:grid-cols-4 gap-4 text-sm">
            <div className="bg-white p-4 rounded">
              <h4 className="font-semibold mb-2 text-indigo-700">1. Конволюция</h4>
              <p>Кернелът се плъзга през изображението с определена стъпка (stride). На всяка позиция се прави поелементно умножение и сумиране.</p>
            </div>
            <div className="bg-white p-4 rounded">
              <h4 className="font-semibold mb-2 text-indigo-700">2. Активация (ReLU)</h4>
              <p>Прилагаме ReLU функция: max(0, x). Отрицателните стойности стават 0, което помага за нелинейност.</p>
            </div>
            <div className="bg-white p-4 rounded">
              <h4 className="font-semibold mb-2 text-orange-700">3. Pooling</h4>
              <p>Max Pooling избира максималната стойност от всяка област. Това намалява размера и прави мрежата устойчива на малки измествания.</p>
            </div>
            <div className="bg-white p-4 rounded">
              <h4 className="font-semibold mb-2 text-purple-700">4. Параметри</h4>
              <p><strong>Stride:</strong> стъпка на плъзгане. <strong>Padding:</strong> добавя нули около изображението за запазване на размера.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CNNVisualization;
