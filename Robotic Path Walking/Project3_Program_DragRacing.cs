
#define LANE2  // Define which lane the car runs on

using System;   
using MonoBrick.EV3; // use this to run the example on the EV3
using System.Threading;
using System.IO;

namespace Application
{
    public static class Program
    {

        const sbyte SPEED = 60;

        static void Main(string[] args)
        {
            try
            {
                var brick = new Brick<ColorSensor, Sensor, GyroSensor, UltrasonicSensor>("COM3");
                brick.Connection.Close();
                brick.Connection.Open();

                brick.Sensor1.Mode = ColorMode.Reflection;
                brick.Sensor3.Mode = GyroMode.AngularVelocity;
                brick.Sensor4.Mode = UltrasonicMode.Centimeter;
                brick.Vehicle.LeftPort = MotorPort.OutA;
                brick.Vehicle.RightPort = MotorPort.OutD;

                brick.MotorA.ResetTacho();  // Tacho count reset
                brick.MotorD.ResetTacho();

#if LANE2       // Ultrasonic sensor points to different directions in two lanes
                brick.MotorC.On(10);  // Right turn
                Thread.Sleep(650);
                brick.MotorC.Off();
#elif LANE1
                brick.MotorC.On(-10);  // Left turn
                Thread.Sleep(700);
                brick.MotorC.Off();
#endif
                File.WriteAllText("data.csv", "");  // Clear the existing csv file

                Console.WriteLine("Press spacebar to begin");

                ConsoleKeyInfo cki;     // Monitoring if spacebar has been pressed
                do
                    cki = Console.ReadKey(true);
                while (cki.Key != ConsoleKey.Spacebar);
                
                Console.Write("\n----- Program Started -----\nPress Q to quit\n");


                // ----- Main program starts ----- //
                int tacho, lightReflection;
                double leftSpeed, rightSpeed;
                bool redLinePassed = false;
                bool blueLinePassed = false;
                bool isRotated = false;

                while ( !blueLinePassed )
                {
                    // Press Q to quit
                    if (Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Q)
                    {
                        brick.Vehicle.Brake();
                        break;
                    }

                    // Update encoder value and reflection value
                    tacho = (brick.MotorA.GetTachoCount() + brick.MotorD.GetTachoCount()) / 2;
                    lightReflection = brick.Sensor1.Read();

                    // left motor speed  = (0.015x + 1)*Current_Speed where x is the distance that exceeds 15cm
                    // right motor speed = (-0.015x + 1)*Current_Speed
                    // The slope of the equation is different in two lanes
                    double distanceToGo = brick.Sensor4.Read() - 15.0;
#if LANE2
                    if (!isRotated) {
                        leftSpeed = ((0.014 * distanceToGo) + 1) * SPEED;
                        rightSpeed = ((-0.014 * distanceToGo) + 1) * SPEED;
                    } else {
                        leftSpeed = ((-0.014 * distanceToGo) + 1) * SPEED;
                        rightSpeed = ((0.014 * distanceToGo) + 1) * SPEED;
                    }
#elif LANE1
                    if (!isRotated) {
                        leftSpeed = ((-0.015 * distanceToGo) + 1) * SPEED;
                        rightSpeed = ((0.015 * distanceToGo) + 1) * SPEED;
                    } else {
                        leftSpeed = ((0.015 * distanceToGo) + 1) * SPEED;
                        rightSpeed = ((-0.015 * distanceToGo) + 1) * SPEED;
                    }
#endif

                    #region Red line not passed: Go to the end
                    if ( !redLinePassed )
                        {
                            if (3350 < tacho && tacho < 5600) // tacho 3350~5600 is color zone -> slow down
                            {
                                brick.MotorA.On( Convert.ToSByte(leftSpeed * 0.13) );
                                brick.MotorD.On( Convert.ToSByte(rightSpeed * 0.13) );
                                writeToCSV(tacho, lightReflection);
                            }
                            else if (tacho > 9700 && lightReflection > 25) // getting closed to the finishing line
                            {
                                redLinePassed = true;
#if LANE2               // Turn back the ultrasonic sensor to face the front
                                brick.MotorC.On(-10);  // Left turn
                                Thread.Sleep(650);
                                brick.MotorC.Off();
#elif LANE1
                                brick.MotorC.On(10);  // Right turn
                                Thread.Sleep(700);
                                brick.MotorC.Off();
#endif
                            } else {
                                brick.MotorA.On( Convert.ToSByte(leftSpeed) );
                                brick.MotorD.On( Convert.ToSByte(rightSpeed) );
                            }
                        }
                        #endregion  // Region: Red line not passed

                 /* Up tp this point, the robot has passed the finishing line. We need to
                  * make some adjustment to the UltraSonic Sensor so that it will face the
                  * front and measure the distance to the wall. The robot will stop ~6 cm
                  * before it and make a beep indicating so.
                  */

                    #region Red line passed, blue line not passed: Rotate 180 and come back
                    else
                    {
                        if (!isRotated) // Rotate the robot
                        {
                            double distanceToFront;
                            do
                                distanceToFront = brick.Sensor4.Read();
                            while (distanceToFront > 9.0);
                            brick.Vehicle.Brake();
                            brick.Beep(100, 500);
                            Thread.Sleep(20000);

                            double angle = 0;
                            brick.Vehicle.SpinRight(10);
                         /* I integrate the angular velocity every 100 ms. So sensor values
                          * should be divided by 10. However, there is latency when sensor
                          * values come back, so divide by 10 doesn't work. I tested several
                          * times and I found 6.3 is an ideal number.
                          */
                            do {
                                angle += brick.Sensor3.Read() / 6.3;
                                Thread.Sleep(100);
                            } while (angle < 170.0);
                            brick.Vehicle.Brake();
#if LANE2
                            brick.MotorC.On(-10);  // Left turn
                            Thread.Sleep(650);
                            brick.MotorC.Off();
#elif LANE1
                            brick.MotorC.On(10);  // Right turn
                            Thread.Sleep(640);
                            brick.MotorC.Off();
#endif
                            isRotated = true;
                            brick.MotorA.ResetTacho();
                            brick.MotorD.ResetTacho();
                        }
                        else // robot has rotated
                        {
                            brick.MotorA.On( Convert.ToSByte(leftSpeed) );
                            brick.MotorD.On( Convert.ToSByte(rightSpeed) );

                            if (tacho > 7700 && lightReflection < 10) { // Cross blue line
                                blueLinePassed = true;
                                brick.Vehicle.Brake();
                            }
                        }
                    }
                    #endregion  // Region: Red line passed, blue line not passed

                } // blue line has passed, end of while loop

                Console.Write("\n\nProgram Ended!\n");
                // ----- Main program ends ------ //


            } // try block
            catch (Exception e) {
                Console.WriteLine("Error: " + e.Message);
                Console.WriteLine("Press any key to end...");
                Console.ReadKey();
            } // catch block
        }

        static void writeToCSV(int column1, int column2)
        {
            string csv = string.Format("{0}, {1}\n", column1, column2);
            File.AppendAllText("data.csv", csv);
        }

    }
}