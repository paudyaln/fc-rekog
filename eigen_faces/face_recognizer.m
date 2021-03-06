function [mean_square_error] = face_recognizer(noOfImage, efaces, mface, weights_mat, test_face)
    mean_square_error = zeros(10,1);
    test_face = histeq(test_face);
    test_face = reshape(test_face, [],1);
    %construct and compare face
    efaces_transpose = transpose(efaces);
    disp(size(efaces_transpose));
    disp(size(test_face));
    disp(size(mface));
    
    weight_testface = efaces_transpose * (double(test_face) - mface);
    for i = 1 : noOfImage*2
        constructed_face = mface + efaces * weights_mat(:,i);
        constructed_face = reshape(constructed_face, [], 1);
        weight_constructed = efaces_transpose * (constructed_face - mface);
        weight_testface = efaces_transpose * (double(test_face) - mface);
        diff = weight_testface - weight_constructed;
        mse = norm(diff(:))^2 / (numel(diff));
        mean_square_error(i) = mse;
%     diff = weight_testface-weights_mat(:,i);
%     mse = (sum(diff.^2))^(1/2);
%     mean_square_error(i) = mse;
    end
    disp('unordered mse');
    disp(mean_square_error);
    disp('sorted mse');
    disp(sort(mean_square_error, 'ascend'));
    [~, mean_square_error] = sort(mean_square_error, 'ascend');
    disp("mse");
    disp(mean_square_error);
    mean_square_error = mean_square_error(1);
end

