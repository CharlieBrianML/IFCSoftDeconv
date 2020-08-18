function result = RL_deconv(image, PSF, iterations)
    % to utilise the conv2 function we must make sure the inputs are double
    image = double(image);
    PSF = double(PSF);
    latent_est = image; % initial estimate, or 0.5*ones(size(image)); 
    PSF_HAT = PSF(end:-1:1,end:-1:1); % spatially reversed psf
    % iterate towards ML estimate for the latent image
    for i= 1:iterations
        est_conv      = conv2(latent_est,PSF,'same');
        relative_blur = image./est_conv;
        error_est     = conv2(relative_blur,PSF_HAT,'same'); 
        latent_est    = latent_est.* error_est;
    end
    result = latent_est;

original = im2double(imread('lena256.png'));
figure; imshow(original); title('Original Image')
hsize=[9 9]; sigma=1;
PSF = fspecial('gaussian', hsize, sigma);
blr = imfilter(original, PSF);
figure; imshow(blr); title('Blurred Image')
res_RL = RL_deconv(blr, PSF, 1000); 
figure; imshow(res_RL); title('Recovered Image')