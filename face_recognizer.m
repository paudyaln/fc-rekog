function [mean_square_error] = face_recognizer(noOfImage, efaces, mface, weights_mat, test_face)
    mean_square_error = zeros(noOfImage,1);
    test_face = reshape(test_face, [],1);
    %construct and compare face
    efaces_transpose = transpose(efaces);
    for i = 1 : noOfImage
        constructed_face = reshape(mface + efaces * weights_mat(:,i), [], 1);
        weight_constructed = efaces_transpose * (constructed_face - mface);
        weight_testface = efaces_transpose * (double(test_face) - mface);
        diff = weight_testface - weight_constructed;
        mse = norm(diff(:))^2 / (numel(diff));
        mean_square_error(i) = mse;
    end

    [~, mean_square_error] = sort(mean_square_error, 'ascend');
    mean_square_error = mean_square_error(1);
end

