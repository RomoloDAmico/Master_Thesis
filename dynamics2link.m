function dx = dynamics2link(~, x, u)
    global g l1 l2 m1 m2 v1 v2

    % Stati
    q1 = x(1); 
    q2 = x(2);
    dq1 = x(3);
    dq2 = x(4);
    dq = [dq1; dq2];

    % Matrici del modello (semplificate)
    M11 = (m1+m2)*l1^2 + m2*l2^2 + 2*m2*l1*l2*cos(q2);
    M12 = m2*l2^2 + m2*l1*l2*cos(q2);
    M21 = M12;
    M22 = m2*l2^2;
    M = [M11 M12; M21 M22];

    C1 = -m2*l1*l2*(2*dq1*dq2 + dq2^2)*sin(q2);
    C2 =  m2*l1*l2*dq1^2*sin(q2);
    C = [C1; C2];

    G1 = (m1+m2)*g*l1*sin(q1) + m2*g*l2*sin(q1+q2);
    G2 = m2*g*l2*sin(q1+q2);
    G = [G1; G2];

    tau = u - [v1*dq1; v2*dq2]; % attriti viscosi
    ddq = M \ (tau - C - G);

    dx = [dq; ddq];
end