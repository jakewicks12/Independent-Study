img = imread('pengiun.jpeg');
imageDouble = im2double(img);

for k = 1:50
    for i = 1:3
        [U,S,V] = svd(imageDouble(:,:,i));
        Uk = U(:,1:k);
        Sk = S(1:k,1:k);
        Vk = V(:,1:k);
        compressedImage(:,:,i) = Uk*Sk*Vk';
    end
    normComp = norm(im2gray(imageDouble)-im2gray(compressedImage));
    imshow(compressedImage);
    title(['Norm = ', num2str(normComp),' for K = ', num2str(k)]);
    drawnow;
    
end
