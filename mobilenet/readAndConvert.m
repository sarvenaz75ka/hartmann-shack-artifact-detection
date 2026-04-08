function I = readAndConvert(filename)
    % خواندن تصویر به صورت grayscale
    I = imread(filename);
    
    % بررسی کنید که تصویر 2 بعدی (grayscale) هست
    if size(I, 3) == 1
        % کپی کردن تصویر grayscale به 3 کانال
        I = repmat(I, [1 1 3]);
    end
        I = single(I) / 255;

end