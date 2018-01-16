function getImage_FER()
       csv_file = 'F:\datasets\face_expression\FER2013\fer2013\fer2013.csv';
       root_folder = 'E:\datasets\FER2013';
       if exist(root_folder,'dir') == 0
           mkdir(root_folder);
       end
       total_num = 35886;
       image_id = 0;
       pixel_size = 48;
       
       csv_fid = fopen(csv_file,'r');
       line_first = fgetl(csv_fid);
       while ~feof(csv_fid)
           line = fgetl(csv_fid);
           image_id = image_id+1;
           
           %% 解析csv每行
           C = strsplit(line,',');
           type_folder = fullfile(root_folder,C{1,3});
           image_folder = fullfile(type_folder,C{1,1});
           if exist(image_folder,'dir')==0
               mkdir(image_folder);
           end
           image_filename = fullfile(image_folder,[num2str(image_id) '.jpg']);
           disp([num2str(image_id) '/' num2str(total_num) ' ' image_filename]);
           
           %% 保存图像
           img_pixel = reshape(str2num(C{1,2}),pixel_size,pixel_size)'/255;
           imshow(img_pixel);
           imwrite(img_pixel,image_filename);
       end
    fclose(csv_fid);
end 


