numOfRuns = 1;
simRunTime = 60;
droneMass = 0.2;
for runNum=1:numOfRuns
    abort = 0;
    myTimer = timer('StartDelay',simRunTime, 'TimerFcn','set_param("ObstacleAvoidanceDemo","SimulationCommand","stop"); abort=1');

    
    Scenario = uavScenario("UpdateRate",100,"ReferenceLocation",[0 0 0]);
    addMesh(Scenario,"cylinder",{[0 0 1] [0 .01]},[0 1 0]);
    InitialPosition = [0 0 -7];
    InitialOrientation = [0 0 0];
    platUAV = uavPlatform("UAV",Scenario, ...
                      "ReferenceFrame","NED", ...
                      "InitialPosition",InitialPosition, ...
                      "InitialOrientation",eul2quat(InitialOrientation));
    updateMesh(platUAV,"quadrotor",{1.2},[0 0 1],eul2tform([0 0 pi]));

    AzimuthResolution = 0.5;      
    ElevationResolution = 2;

    MaxRange = 7;
    AzimuthLimits = [-179 179];
    ElevationLimits = [-15 15];

    LidarModel = uavLidarPointCloudGenerator("UpdateRate",10, ...
                                         "MaxRange",MaxRange, ...
                                         "RangeAccuracy",3, ...
                                         "AzimuthResolution",AzimuthResolution, ...
                                         "ElevationResolution",ElevationResolution, ...
                                         "AzimuthLimits",AzimuthLimits, ...
                                         "ElevationLimits",ElevationLimits, ...                                       
                                         "HasOrganizedOutput",true);

    uavSensor("Lidar",platUAV,LidarModel, ...
          "MountingLocation",[0 0 -0.4], ...
          "MountingAngles",[0 0 180]);

    omap3D =  occupancyMap3D;
    mapWidth = 50;
    mapLength = 50;
    numberOfObstacles = randi([3 6],1);
    obstacleNumber = 1;
    while obstacleNumber <= numberOfObstacles
        width = randi([3 15],1);                 % The largest integer in the sample intervals for obtaining width, length and height                                                     
        length = randi([3 15],1);                % can be changed as necessary to create different occupancy maps.
        height = randi([5 20],1);
        xPosition = randi([0 mapWidth-width],1);
        yPosition = randi([0 mapLength-length],1);
        
        [xObstacle,yObstacle,zObstacle] = meshgrid(xPosition:xPosition+width,yPosition:yPosition+length,0:height);
        xyzObstacles = [xObstacle(:) yObstacle(:) zObstacle(:)];
        
        checkIntersection = false;
        for i = 1:size(xyzObstacles,1)
            if checkOccupancy(omap3D,xyzObstacles(i,:)) == 1
                checkIntersection = true;
                break
            end
        end
        if checkIntersection
            continue
        end
        
        setOccupancy(omap3D,xyzObstacles,1)
    
        addMesh(Scenario,"polygon", {[xPosition+1 yPosition+1;xPosition+width-1 yPosition+1;xPosition+width-1 yPosition+length-1;xPosition+1 yPosition+length-1],[0 height]},0.651*ones(1,3))
        
        obstacleNumber = obstacleNumber + 1;
    end
    [xGround,yGround,zGround] = meshgrid(0:mapWidth,0:mapLength,0);
    xyzGround = [xGround(:) yGround(:) zGround(:)];
    setOccupancy(omap3D,xyzGround,1)

    Waypoints = [InitialPosition];

    numberOfWaypoints = randi([3 6],1);
    waypointNumber = 1;
    while waypointNumber <= numberOfWaypoints
        x = randi([3 45],1);                 % The largest integer in the sample intervals for obtaining width, length and height                                                     
        y = randi([3 45],1);                % can be changed as necessary to create different occupancy maps.
        z = randi([3 25],1);
        [xWP,yWP,zWP] = meshgrid(x-1:x+1,y-1:y+1,z-1:z+1);
        xyzWP = [xWP(:) yWP(:) zWP(:)];
        
        checkIntersection = false;
        for i = 1:size(xyzWP,1)
            if checkOccupancy(omap3D,xyzWP(i,:)) == 1
                checkIntersection = true;
                break
            end
        end
        if checkIntersection
            continue
        end
        
        setOccupancy(omap3D,xyzWP,1)
    
        addMesh(Scenario,"cylinder",{[x y 1] [z+1 z+1.1]},[1 0 0]);    
        waypointPos = [y x -z];
        Waypoints = vertcat(Waypoints,waypointPos);
    
        waypointNumber = waypointNumber + 1;
    end

    % Proportional Gains
    Px = 6;
    Py = 6;
    Pz = 6.5;
    
    % Derivative Gains
    Dx = 1.5;
    Dy = 1.5;
    Dz = 2.5;
    
    % Integral Gains
    Ix = 0;
    Iy = 0;
    Iz = 0;
    
    % Filter Coefficients
    Nx = 10;
    Ny = 10;
    Nz = 14.4947065605712; 

    UAVSampleTime = 0.01;
    Gravity = 9.81;
    DroneMass = droneMass;

    start(myTimer);
    out = sim("ObstacleAvoidanceDemo.slx");
    stop(myTimer);

    rollData = timetable2table(ts2timetable(out.roll));
    rollData.Properties.VariableNames = {'Time','Roll CS'};
    pitchData = timetable2table(ts2timetable(out.pitch));
    pitchData.Properties.VariableNames = {'Time','Pitch CS'};
    thrustData = timetable2table(ts2timetable(out.thrust));
    thrustData.Properties.VariableNames = {'Time','Thrust CS'};
    yawData = timetable2table(ts2timetable(out.yaw));
    yawData.Properties.VariableNames = {'Time','Yaw CS'};
    desiredPositionData = array2table(squeeze(out.desiredPosition.data)');
    desiredYawData = timetable2table(ts2timetable(out.desiredYaw));
    
    positionData = array2table(squeeze(out.trajectoryPoints)');

    WorldPosition = timetable2table(ts2timetable(out.UAVState.WorldPosition));
    Thrust = timetable2table(ts2timetable(out.UAVState.Thrust));
    BodyAngularRateRPY = timetable2table(ts2timetable(out.UAVState.BodyAngularRateRPY));  
    EulerZYX = timetable2table(ts2timetable(out.UAVState.EulerZYX));
    WorldVelocity = timetable2table(ts2timetable(out.UAVState.WorldVelocity));

    positionData.Properties.VariableNames = {'x','y','z'};
    
    desiredPositionData.Properties.VariableNames = {'desired x','desired y','desired z'};
    desiredYawData.Properties.VariableNames = {'Time','desired yaw'};
    
    testData = horzcat(rollData,pitchData(:,2),yawData(:,2),thrustData(:,2),positionData,desiredPositionData,desiredYawData(:,2), WorldPosition(:,2),Thrust(:,2),BodyAngularRateRPY(:,2),EulerZYX(:,2),WorldVelocity(:,2));

    if abort == 1
        writetable(testData,"timeOut_dataset"+num2str(runNum)+".csv",'Delimiter',',','QuoteStrings',true);
    else
        writetable(testData,"dataset"+num2str(runNum)+".csv",'Delimiter',',','QuoteStrings',true);
    end
end

