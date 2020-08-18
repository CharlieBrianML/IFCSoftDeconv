using Images, TestImages, Deconvolution, FFTW, ZernikePolynomials, ImageView, Gtk

img = channelview(testimage("cameraman"))

# model of lens aberration
blurring = evaluateZernike(LinRange(-16,16,512), [12, 4, 0], [1.0, -1.0, 2.0], index=:OSA)
blurring = fftshift(blurring)
blurring = blurring ./ sum(blurring)

blurred_img = fft(img) .* fft(blurring) |> ifft |> real

@time restored_img = lucy(blurred_img, blurring, iterations=10)

ImageView.imshow(img)
imshow(blurring)
imshow(blurred_img)
imshow(restored_img)
#Gtk.showall(gui["window"])

println("Hola")
