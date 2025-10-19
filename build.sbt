ThisBuild / version := "0.1.0"

ThisBuild / scalaVersion := "3.6.3"

lazy val root = (project in file("."))
  .settings(
    name := "pytorch_on_scala3_lesson"
  )


libraryDependencies +=   "io.github.mullerhai" % "storch_core_3" % "0.7.3-1.5.12"
libraryDependencies +=   "io.github.mullerhai" % "storch-gpu-adapter_3" % "0.1.3-1.5.12"
// https://mvnrepository.com/artifact/org.bytedeco/cuda-platform-redist
//libraryDependencies += "org.bytedeco" % "cuda-platform-redist" % "12.9-9.10-1.5.12"