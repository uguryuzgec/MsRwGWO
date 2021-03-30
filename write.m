function write(a,name)
        fid   = fopen(name,'w');
        [m,n] = size(a);
     for i=1:m
        for j=1:n
            if j==n
               fprintf(fid,'%2.3e\n',a(i,j));
            else
               fprintf(fid,'%2.3e\t',a(i,j));
            end
        end    
     end
    fclose(fid);
    judge    = exist('Result_data');
    if judge ~= 7
        system('mkdir Result_data');
    end
    file_path   = strcat(cd,'\Result_data');
    movefile(name,file_path);
end