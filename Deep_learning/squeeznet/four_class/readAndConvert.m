function I = readAndConvert(filename)
    % خواندن تصویر
    I = imread(filename);
    
    % --- 1. تضمین 3 کانال (مورد نیاز SqueezeNet) ---
    if size(I, 3) == 1
        % کپی کردن تصویر grayscale به 3 کانال (R, G, B)
        I = repmat(I, [1 1 3]);
    end
    
    % --- 2. تبدیل نوع داده به single ---
    I = single(I);
    
    % --- 3. استانداردسازی ImageNet (ضروری برای Transfer Learning) ---
    
    % میانگین و انحراف معیار استاندارد ImageNet برای کانال‌های R, G, B
    % مقادیر استاندارد نرمال‌شده (0 تا 1) هستند که در 255 ضرب شده‌اند.
    mu = [0.485, 0.456, 0.406] * 255; 
    sigma = [0.229, 0.224, 0.225] * 255;
    
    % اعمال فرمول استانداردسازی: I_norm = (I - mu) / sigma
    for i = 1:3
        % اعمال میانگین و انحراف معیار خاص هر کانال
        I(:,:,i) = (I(:,:,i) - mu(i)) / sigma(i);
    end
    
    % نکته: تغییر اندازه (Resize) توسط augmentedImageDatastore انجام خواهد شد.
end