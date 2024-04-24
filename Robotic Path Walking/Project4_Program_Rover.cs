using System;   
using MonoBrick.EV3; // use this to run the example on the EV3
using System.Threading;
using System.IO;

// Use MyBrick to eliminate writing out the long messy command
using MyBrick = MonoBrick.EV3.Brick<MonoBrick.EV3.ColorSensor,
                                    MonoBrick.EV3.Sensor,
                                    MonoBrick.EV3.GyroSensor,
                                    MonoBrick.EV3.UltrasonicSensor>;
namespace Application
{
    public enum Direction {
        North, East, South, West
    };


    public static class Position
    {
        // A struct named Record to keep track of data in the grid that the robot has walked
        private struct Record {
            public string robotName;
            public Direction direction;
            public int row;
            public int column;
        };

        private static Record[,] positionRecord = new Record[3,5];

        public static void update(int row, int column, Direction direction)
        {
            positionRecord[row, column].robotName = "CHARMANDER";
            positionRecord[row, column].direction = direction;
            positionRecord[row, column].row = row;
            positionRecord[row, column].column= column;
        }
    }
    
    class Program
    {
        const sbyte SPEED = 10;
        static Direction currDirection = Direction.North;
        static int row;
        static int column;

        static int originRow;
        static int originColumn;
        
        private static void Forward(MyBrick brick)
        {
            int currTacho = brick.MotorA.GetTachoCount();
            do
                brick.Vehicle.Forward(SPEED);
            while (brick.MotorA.GetTachoCount() < currTacho + 200);
            int lightReflection;
            int count = 0;
            do {
                if (count % 70 == 0) {
                    brick.MotorA.On(SPEED);
                    brick.MotorD.Off();
                } else
                    brick.Vehicle.Forward(SPEED);
                lightReflection = brick.Sensor1.Read();
                Thread.Sleep(50);
                count++;
            }  while (lightReflection > 3);
            brick.Vehicle.Brake();

            switch (currDirection) {
                case Direction.North: row++; break;
                case Direction.South: row--; break;
                case Direction.East: column++; break;
                case Direction.West: column--; break;
                default: break;
            }
            //Position.update(row, column, currDirection);
        }

        private static void Backward(MyBrick brick, bool updatePosition)
        {
            int currTacho = brick.MotorA.GetTachoCount();
            do
                brick.Vehicle.Forward(-SPEED);
            while (brick.MotorA.GetTachoCount() < currTacho + 200);
            int lightReflection;
            do
            {
                brick.Vehicle.Forward(-SPEED);
                lightReflection = brick.Sensor1.Read();
                Thread.Sleep(50);
            } while (lightReflection > 3);
            brick.Vehicle.Brake();

            if (updatePosition)
                switch (currDirection)
                {
                    case Direction.North: row--; break;
                    case Direction.South: row++; break;
                    case Direction.East: column--; break;
                    case Direction.West: column++; break;
                    default: break;
                }
            //Position.update(row, column, currDirection);
        }

        private static void RotateCW90(MyBrick brick)
        {
            int leftTacho = brick.MotorA.GetTachoCount();
            do {
                brick.MotorA.On(SPEED);
                brick.MotorD.On(-SPEED);
            } while (brick.MotorA.GetTachoCount() < leftTacho + 160);
            brick.Vehicle.Brake();

            switch (currDirection) {
                case Direction.North: currDirection = Direction.East; break;
                case Direction.East: currDirection = Direction.South; break;
                case Direction.South: currDirection = Direction.West; break;
                case Direction.West: currDirection = Direction.North; break;
                default: break;
            }
        }

        private static void RotateCounterCW90(MyBrick brick)
        {
            int leftTacho = brick.MotorA.GetTachoCount();
            do {
                brick.MotorA.On(-SPEED);
                brick.MotorD.On(SPEED);
            } while (brick.MotorA.GetTachoCount() > leftTacho - 160);
            brick.Vehicle.Brake();

            switch (currDirection) {
                case Direction.North: currDirection = Direction.West; break;
                case Direction.East: currDirection = Direction.North; break;
                case Direction.South: currDirection = Direction.East; break;
                case Direction.West: currDirection = Direction.South; break;
                default: break;
            }
        }

        private static void RotateCounterCW180(MyBrick brick) {
            RotateCounterCW90(brick);
            RotateCounterCW90(brick);
        }

        static void Main(string[] args)
        {
            try {

                var brick = new MyBrick("COM3");
                brick.Connection.Close();
                brick.Connection.Open();

                brick.Sensor1.Mode = ColorMode.Reflection;
                brick.Sensor4.Mode = UltrasonicMode.Centimeter;
                brick.Vehicle.LeftPort = MotorPort.OutA;
                brick.Vehicle.RightPort = MotorPort.OutD;

                int lightReflection;
                bool originCloseToRightWall;

                #region Go to the nearest intersection
                Console.WriteLine("Go to the nearest intersection.");
                DateTime start = DateTime.Now;

                // Check if robot is too close to wall in front.
                if (brick.Sensor4.Read() > 15.0)
                    brick.Vehicle.Forward(SPEED);
                else
                    brick.Vehicle.Forward(-SPEED);
                do
                {
                    lightReflection = brick.Sensor1.Read();
                    Thread.Sleep(30);
                } while (lightReflection > 3);

                // Check if robot is too close to wall besides it.
                RotateCW90(brick);
                if (brick.Sensor4.Read() > 15.0)
                {
                    Forward(brick);
                    originCloseToRightWall = false;
                }
                else
                {
                    RotateCounterCW180(brick);
                    Forward(brick);
                    originCloseToRightWall = true;
                }
                #endregion

                #region Calculate original position
                Console.WriteLine("Start calculating position.");
                if (!originCloseToRightWall)
                    RotateCounterCW180(brick);
                originColumn = (int)Math.Round(brick.Sensor4.Read() / 20);
                RotateCounterCW90(brick);
                originRow = (int)Math.Round(brick.Sensor4.Read() / 20);

                DateTime stop = DateTime.Now;
                Console.WriteLine("Position: Row: {0} \t Column: {1}", originRow, originColumn);
                Console.WriteLine("Time elapsed: {0}", stop - start);
                brick.Beep(100, 500);
                Console.ReadKey(true);

                column = originColumn;
                row = originRow;
                #endregion

                #region Object Searching
                // Search for object.
                Console.WriteLine("Search for object.");

                double distance;
                double noObjectDistance;
                bool found = false;

                start = DateTime.Now;

                // Check if object is already on the line while going to one side of wall.
                // Front
                Console.WriteLine("Checking front");
                distance = brick.Sensor4.Read();
                noObjectDistance = row * 20.32;
                if (distance < noObjectDistance - 6.5)
                {
                    Console.WriteLine("Found!");
                    found = true;
                }
                // Left hand side
                if (!found)
                {
                    RotateCW90(brick);
                    Console.WriteLine("Checking left");
                    distance = brick.Sensor4.Read();
                    noObjectDistance = column * 20.32;
                    if (distance < noObjectDistance - 6.5)
                    {
                        Console.WriteLine("Found!");
                        found = true;
                    }
                }
                // Back side
                if (!found)
                {
                    RotateCW90(brick);
                    Console.WriteLine("Checking back");
                    distance = brick.Sensor4.Read();
                    noObjectDistance = 80 - (row * 20.32);
                    if (distance < noObjectDistance - 6.5)
                    {
                        Console.WriteLine("Found!");
                        found = true;
                    }
                }
                // Right hand side
                if (!found)
                {
                    RotateCW90(brick);
                    Console.WriteLine("Checking right");
                    distance = brick.Sensor4.Read();
                    noObjectDistance = 120 - (column * 20.32);
                    if (distance < noObjectDistance - 6.5)
                    {
                        Console.WriteLine("Found!");
                        found = true;
                    }
                }

                if (!found)
                {
                    // Going to the right-end base line
                    do
                    {
                        Console.WriteLine("Forward to right base line");
                        Forward(brick);
                        distance = brick.Sensor4.Read();
                    } while (distance > 22.0);

                    switch (originRow)
                    {
                        case 1:
                            RotateCounterCW90(brick);
                            distance = brick.Sensor4.Read();
                            noObjectDistance = 60.96;
                            if (distance < noObjectDistance - 6.5)
                            {
                                Console.WriteLine("Found!");
                                found = true;
                            }
                            Console.WriteLine("Current intersection cleared!");
                            // Search in row 2 and 3
                            for (int i = 0; i < 2 && !found; i++)
                            {
                                Forward(brick);
                                RotateCounterCW90(brick);
                                distance = brick.Sensor4.Read();
                                noObjectDistance = 101.6;
                                if (distance < noObjectDistance - 6.5)
                                {
                                    Console.WriteLine("Found!");
                                    found = true;
                                }
                                if (i == 0)
                                    RotateCW90(brick);
                            }
                            break;

                        case 2:
                            RotateCounterCW90(brick);
                            distance = brick.Sensor4.Read();
                            noObjectDistance = 40.64;
                            if (distance < noObjectDistance - 6.5)
                            {
                                Console.WriteLine("Found!");
                                found = true;
                            }
                            // Search in row 3
                            if (!found)
                            {
                                Forward(brick);
                                RotateCounterCW90(brick);
                                distance = brick.Sensor4.Read();
                                noObjectDistance = 101.6;
                                if (distance < noObjectDistance - 6.5)
                                {
                                    Console.WriteLine("Found!");
                                    found = true;
                                }
                            }
                            // Search in row 1
                            if (!found)
                            {
                                RotateCounterCW90(brick);
                                distance = brick.Sensor4.Read();
                                noObjectDistance = 60.96;
                                if (distance < noObjectDistance - 6.5)
                                {
                                    Console.WriteLine("Found!");
                                    found = true;
                                }
                            }
                            if (!found)
                            {
                                Forward(brick);
                                Forward(brick);
                                RotateCW90(brick);
                                distance = brick.Sensor4.Read();
                                noObjectDistance = 60.96;
                                if (distance < noObjectDistance - 6.5)
                                {
                                    Console.WriteLine("Found!");
                                    found = true;
                                }
                            }
                            break;

                        case 3:
                            RotateCW90(brick);
                            distance = brick.Sensor4.Read();
                            noObjectDistance = 60.96;
                            if (distance < noObjectDistance - 6.5)
                            {
                                Console.WriteLine("Found!");
                                found = true;
                            }
                            Console.WriteLine("Current intersection cleared!");
                            // Search in row 2 and 3
                            for (int i = 0; i < 2 && !found; i++)
                            {
                                Forward(brick);
                                RotateCW90(brick);
                                distance = brick.Sensor4.Read();
                                noObjectDistance = 101.6;
                                if (distance < noObjectDistance - 6.5)
                                {
                                    Console.WriteLine("Found!");
                                    found = true;
                                }
                                if (i == 0)
                                    RotateCounterCW90(brick);
                            }
                            break;

                        default:
                            break;
                    }
                }
                #endregion

                #region Go to the object
                Console.WriteLine("Going to object.");
                do
                {
                    Forward(brick);
                    distance = brick.Sensor4.Read();
                } while (distance > 22);

                int endRow = row;
                int endColumn = column;

                do
                {
                    brick.Vehicle.Forward(SPEED);
                    distance = brick.Sensor4.Read();
                } while (distance > 3);
                brick.Vehicle.Brake();

                stop = DateTime.Now;
                Console.WriteLine("Time elapsed for searching and reaching object: {0}", stop - start);

                brick.MotorC.On(SPEED, 130, false);
                Thread.Sleep(1000);

                start = DateTime.Now;
                switch (currDirection)
                {
                    case Direction.North:
                        Backward(brick, false);
                        if (endRow == originRow)
                            RotateCounterCW180(brick);
                        else
                        {
                            RotateCounterCW90(brick);
                            for (int i = 0; i < Math.Abs(endColumn - originColumn); i++)
                                Forward(brick);
                            RotateCounterCW90(brick);
                        }
                        break;

                    case Direction.West:
                        RotateCounterCW90(brick);
                        for (int i = 0; i < Math.Abs(endRow - originRow); i++)
                            Forward(brick);
                        RotateCounterCW90(brick);
                        for (int i = 0; i < Math.Abs(endColumn - originColumn); i++)
                            Forward(brick);
                        RotateCW90(brick);
                        break;

                    case Direction.East:
                        RotateCounterCW180(brick);
                        for (int i = 0; i < Math.Abs(endColumn - originColumn); i++)
                            Forward(brick);
                        RotateCounterCW90(brick);
                        break;

                    case Direction.South:
                        Backward(brick, false);
                        if (endRow == originRow)
                            RotateCounterCW180(brick);
                        else
                        {
                            RotateCW90(brick);
                            for (int i = 0; i < Math.Abs(endColumn - originColumn); i++)
                                Forward(brick);
                            RotateCW90(brick);
                        }
                        break;

                    default:
                        break;
                }
                stop = DateTime.Now;
                Console.WriteLine("Time elapsed for moving the object back to origin: {0}", stop - start);
                #endregion

                brick.Connection.Close();
            } // try block
            catch (Exception e) {
                Console.WriteLine("Error: " + e.Message);
                Console.WriteLine("Press any key to end...");
                Console.ReadKey();
            } // catch block
        } // Main function
    }
}